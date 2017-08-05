package predict

import (
	"bufio"
	"image"
	"os"
	"path/filepath"
	"strings"

	context "golang.org/x/net/context"

	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	"github.com/anthonynsimon/bild/parallel"
	"github.com/anthonynsimon/bild/transform"
	opentracing "github.com/opentracing/opentracing-go"
	"github.com/pkg/errors"
	"github.com/rai-project/caffe"
	"github.com/rai-project/dlframework"
	common "github.com/rai-project/dlframework/framework/predict"
	"github.com/rai-project/downloadmanager"
	gocaffe "github.com/rai-project/go-caffe"
)

type ImagePredictor struct {
	common.ImagePredictor
	workDir   string
	features  []string
	predictor *gocaffe.Predictor
}

func New(model dlframework.ModelManifest) (common.Predictor, error) {
	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}
	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}
	return newImagePredictor(model)
}

func newImagePredictor(model dlframework.ModelManifest) (*ImagePredictor, error) {
	framework, err := model.ResolveFramework()
	if err != nil {
		return nil, err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return nil, err
	}

	ip := &ImagePredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
			},
		},
		workDir: workDir,
	}

	return ip, nil
}

func (p *ImagePredictor) GetWeightsUrl() string {
	model := p.Model
	if model.GetModel().GetIsArchive() {
		return model.GetModel().GetBaseUrl()
	}
	baseURL := ""
	if model.GetModel().GetBaseUrl() != "" {
		baseURL = strings.TrimSuffix(model.GetModel().GetBaseUrl(), "/") + "/"
	}
	return baseURL + model.GetModel().GetWeightsPath()
}

func (p *ImagePredictor) GetGraphUrl() string {
	model := p.Model
	if model.GetModel().GetIsArchive() {
		return model.GetModel().GetBaseUrl()
	}
	baseURL := ""
	if model.GetModel().GetBaseUrl() != "" {
		baseURL = strings.TrimSuffix(model.GetModel().GetBaseUrl(), "/") + "/"
	}
	return baseURL + model.GetModel().GetGraphPath()
}

func (p *ImagePredictor) GetFeaturesUrl() string {
	model := p.Model
	params := model.GetOutput().GetParameters()
	pfeats, ok := params["features_url"]
	if !ok {
		return ""
	}
	return pfeats.Value
}

func (p *ImagePredictor) GetGraphPath() string {
	model := p.Model
	graphPath := filepath.Base(model.GetModel().GetGraphPath())
	return filepath.Join(p.workDir, graphPath)
}

func (p *ImagePredictor) GetWeightsPath() string {
	model := p.Model
	graphPath := filepath.Base(model.GetModel().GetWeightsPath())
	return filepath.Join(p.workDir, graphPath)
}

func (p *ImagePredictor) GetFeaturesPath() string {
	model := p.Model
	return filepath.Join(p.workDir, model.GetName()+".features")
}

func (p *ImagePredictor) GetMeanPath() string {
	model := p.Model
	return filepath.Join(p.workDir, model.GetName()+".mean")
}

func (p *ImagePredictor) Preprocess(ctx context.Context, input interface{}) (interface{}, error) {

	if span, newCtx := opentracing.StartSpanFromContext(ctx, "Preprocess"); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

	img, ok := input.(image.Image)
	if !ok {
		return nil, errors.New("expecting an image input")
	}

	imageDims, err := p.GetImageDimensions()
	if err != nil {
		return nil, err
	}
	img = transform.Resize(img, int(imageDims[2]), int(imageDims[3]), transform.Linear)

	b := img.Bounds()
	height := b.Max.Y - b.Min.Y // image height
	width := b.Max.X - b.Min.X  // image width

	imageSize := 3 * height * width

	meanImage, err := p.GetMeanImage(ctx, p.readBlobfromURL)
	if err != nil && len(meanImage) != imageSize {
		meanImage = make([]float32, imageSize)
	}

	if len(meanImage) != imageSize {
		lenMeanImage := len(meanImage)
		resizedMeanImage := make([]float32, imageSize)
		for ii := 0; ii < imageSize; ii++ {
			resizedMeanImage[ii] = meanImage[ii%lenMeanImage]
		}
		meanImage = resizedMeanImage
	}

	if len(meanImage) != imageSize {
		return nil, errors.Errorf("mean image size mismatch. %v != %v ", len(meanImage), imageSize)
	}

	res := make([]float32, imageSize)
	parallel.Line(height, func(start, end int) {
		w := width
		h := height

		for y := start; y < end; y++ {
			for x := 0; x < width; x++ {
				r, g, b, _ := img.At(x+b.Min.X, y+b.Min.Y).RGBA()
				res[y*w+x] = float32(r>>8) - meanImage[y*w+x]
				res[w*h+y*w+x] = float32(g>>8) - meanImage[w*h+y*w+x]
				res[2*w*h+y*w+x] = float32(b>>8) - meanImage[2*w*h+y*w+x]
			}
		}

	})
	return res, nil
}

func (p *ImagePredictor) readBlobfromURL(ctx context.Context, url string) ([]float32, error) {
	targetPath := filepath.Join(p.workDir, "mean.binaryproto")

	fileName, err := downloadmanager.DownloadFile(ctx, url, targetPath)
	if err != nil {
		return nil, err
	}
	blob, err := caffe.ReadBlob(fileName)
	if err != nil {
		return nil, err
	}

	return blob.Data, nil
}

func (p *ImagePredictor) Download(ctx context.Context) error {

	if span, newCtx := opentracing.StartSpanFromContext(ctx, "DownloadingModel"); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

	if _, err := downloadmanager.DownloadFile(ctx, p.GetGraphUrl(), p.GetGraphPath()); err != nil {
		return err
	}
	if _, err := downloadmanager.DownloadFile(ctx, p.GetWeightsUrl(), p.GetWeightsPath()); err != nil {
		return err
	}
	if _, err := downloadmanager.DownloadFile(ctx, p.GetFeaturesUrl(), p.GetFeaturesPath()); err != nil {
		return err
	}
	return nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	if p.predictor != nil {
		return nil
	}

	var features []string
	f, err := os.Open(p.GetFeaturesPath())
	if err != nil {
		return errors.Wrapf(err, "cannot read %s", p.GetFeaturesPath())
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		features = append(features, line)
	}

	p.features = features

	inputDims, err := p.GetImageDimensions()
	if err != nil {
		return err
	}
	modelInputShape := make([]uint32, len(inputDims))
	for ii, v := range inputDims {
		modelInputShape[ii] = uint32(v)
	}

	pred, err := gocaffe.New(p.GetGraphPath(), p.GetWeightsPath())
	if err != nil {
		return err
	}
	p.predictor = pred

	return nil
}

func (p *ImagePredictor) Predict(ctx context.Context, input interface{}) (*dlframework.PredictionFeatures, error) {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "Predict"); span != nil {
		ctx = newCtx
		defer span.Finish()
	}
	if err := p.loadPredictor(ctx); err != nil {
		return nil, err
	}

	imageData, ok := input.([]float32)
	if !ok {
		return nil, errors.New("expecting []float32 input in predict function")
	}

	predictions, err := p.predictor.Predict(imageData)
	if err != nil {
		return nil, err
	}

	rprobs := make([]*dlframework.PredictionFeature, len(predictions))
	for ii, pred := range predictions {
		rprobs[ii] = &dlframework.PredictionFeature{
			Index:       int64(pred.Index),
			Name:        p.features[pred.Index],
			Probability: pred.Probability,
		}
	}
	res := dlframework.PredictionFeatures(rprobs)
	res.Sort()

	return &res, nil
}

func (p *ImagePredictor) Close() error {
	if p.predictor != nil {
		p.predictor.Close()
	}
	return nil
}
