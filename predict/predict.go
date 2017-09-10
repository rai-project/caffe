package predict

import (
	"bufio"
	"os"
	"strings"

	context "golang.org/x/net/context"

	opentracing "github.com/opentracing/opentracing-go"
	"github.com/pkg/errors"
	"github.com/rai-project/caffe"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	common "github.com/rai-project/dlframework/framework/predict"
	"github.com/rai-project/downloadmanager"
	gocaffe "github.com/rai-project/go-caffe"
	"github.com/rai-project/image/types"
)

type ImagePredictor struct {
	common.ImagePredictor
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

	predictor := new(ImagePredictor)

	return predictor.Load(context.Background(), model)
}

func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest) (common.Predictor, error) {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "Load"); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

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
			WorkDir: workDir,
		},
	}

	ip.download(ctx)
	ip.loadPredictor(ctx)

	return ip, nil
}

func (p *ImagePredictor) download(ctx context.Context) error {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "Download"); span != nil {
		span.SetTag("graph_url", p.GetGraphUrl())
		span.SetTag("traget_graph_file", p.GetGraphPath())
		span.SetTag("weights_url", p.GetWeightsUrl())
		span.SetTag("traget_weights_file", p.GetWeightsPath())
		span.SetTag("feature_url", p.GetFeaturesUrl())
		span.SetTag("traget_feature_file", p.GetFeaturesPath())
		ctx = newCtx
		defer span.Finish()
	}

	model := p.Model
	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		_, err := downloadmanager.DownloadInto(baseURL, p.WorkDir)
		if err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
		return nil
	}

	if _, err := downloadmanager.DownloadFile(p.GetGraphUrl(), p.GetGraphPath()); err != nil {
		return err
	}

	if _, err := downloadmanager.DownloadFile(p.GetWeightsUrl(), p.GetWeightsPath()); err != nil {
		return err
	}

	if _, err := downloadmanager.DownloadFile(p.GetFeaturesUrl(), p.GetFeaturesPath()); err != nil {
		return err
	}

	return nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "LoadPredictor"); span != nil {
		ctx = newCtx
		defer span.Finish()
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

	pred, err := gocaffe.New(p.GetGraphPath(), p.GetWeightsPath())
	if err != nil {
		return err
	}
	p.predictor = pred

	return nil
}

<<<<<<< HEAD
func (p *ImagePredictor) PreprocessOptions(ctx context.Context) (predict.PreprocessOptions, error) {
	mean, err := p.GetMeanImage()
	if err != nil {
		return predict.PreprocessorOptions{}, err
=======
func (p *ImagePredictor) PreprocessOptions(ctx context.Context) (common.PreprocessOptions, error) {
	mean, err := p.GetMeanImage()
	if err != nil {
		return common.PreprocessOptions{}, err
>>>>>>> 9a2c07a0b8174dfba86d64ff9542869c578d9e83
	}

	scale, err := p.GetScale()
	if err != nil {
<<<<<<< HEAD
		return predict.PreprocessorOptions{}, err
	}

	imageDims, err := p.GetImageDimensions()
	if err != nil {
		return predict.PreprocessorOptions{}, err
	}

	return PreprocessorOptions{
		MeanImage:  mean,
		Scale:      scale,
		Size:       []int{int(imageDims[2]), int(imageDims[3])},
		ColorSpace: types.BGRMode,
=======
		return common.PreprocessOptions{}, err
	}

	imageDims, err := p.GetImageDimensions()
	if err != nil {
		return common.PreprocessOptions{}, err
>>>>>>> 9a2c07a0b8174dfba86d64ff9542869c578d9e83
	}
}

<<<<<<< HEAD
=======
	return common.PreprocessOptions{
		MeanImage: mean,
		Scale:     scale,
		Size:      []int{int(imageDims[2]), int(imageDims[3])},
		ColorMode: types.BGRMode,
	}, nil
}

>>>>>>> 9a2c07a0b8174dfba86d64ff9542869c578d9e83
func (p *ImagePredictor) Predict(ctx context.Context, data []float32) (dlframework.Features, error) {
	if span, newCtx := opentracing.StartSpanFromContext(ctx, "Predict"); span != nil {
		ctx = newCtx
		defer span.Finish()
	}

	predictions, err := p.predictor.Predict(data)
	if err != nil {
		return nil, err
	}

	rprobs := make([]*dlframework.Feature, len(predictions))
	for ii, pred := range predictions {
		rprobs[ii] = &dlframework.Feature{
			Index:       int64(pred.Index),
			Name:        p.features[pred.Index],
			Probability: pred.Probability,
		}
	}
	res := dlframework.Features(rprobs)

	return res, nil
}

func (p *ImagePredictor) Reset(ctx context.Context) error {

	return nil
}

func (p *ImagePredictor) Close() error {
	if p.predictor != nil {
		p.predictor.Close()
	}
	return nil
}

func init() {
	config.AfterInit(func() {
		framework := caffe.FrameworkManifest
		agent.AddPredictor(framework, &ImagePredictor{
			ImagePredictor: common.ImagePredictor{
				Base: common.Base{
					Framework: framework,
				},
			},
		})
	})
}
