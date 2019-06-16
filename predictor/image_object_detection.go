package predictor

import (
	"context"
	"strings"

	"github.com/k0kubun/pp"
	"github.com/pkg/errors"
	"github.com/rai-project/caffe"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/agent"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	"github.com/rai-project/tracer"
	"github.com/rai-project/tracer/ctimer"
	gotensor "gorgonia.org/tensor"
)

type ImageObjectDetectionPredictor struct {
	*ImagePredictor
	boxesLayerIndex         int
	probabilitiesLayerIndex int
	classesLayerIndex       int
	boxes                   interface{}
	probabilities           interface{}
	classes                 interface{}
}

// New ...
func NewImageObjectDetectionPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
	ctx := context.Background()
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "new_predictor")
	defer span.Finish()

	modelInputs := model.GetInputs()
	if len(modelInputs) != 1 {
		return nil, errors.New("number of inputs not supported")
	}

	firstInputType := modelInputs[0].GetType()
	if strings.ToLower(firstInputType) != "image" {
		return nil, errors.New("input type not supported")
	}

	predictor := new(ImageObjectDetectionPredictor)
	return predictor.Load(ctx, model, opts...)
}

func (self *ImageObjectDetectionPredictor) Load(ctx context.Context, modelManifest dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {

	pred, err := self.ImagePredictor.Load(ctx, modelManifest, opts...)
	if err != nil {
		return nil, err
	}
	p := &ImageObjectDetectionPredictor{
		ImagePredictor: pred,
	}

	p.probabilitiesLayerIndex, err = p.GetOutputLayerIndex("probabilities_layer")
	p.boxesLayerIndex, err = p.GetOutputLayerIndex("boxes_layer")
	p.classesLayerIndex, err = p.GetOutputLayerIndex("classes_layer")

	if err != nil {
		return nil, errors.Wrap(err, "failed to get the probabilities layer index")
	}

	return p, nil
}

// Predict ...
func (p *ImageObjectDetectionPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "predict")
	defer span.Finish()

	if p.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		err := p.predictor.StartProfiling("caffe", "c_predict")
		if err != nil {
			log.WithError(err).WithField("framework", "caffe").Error("unable to start framework profiling")
		} else {
			defer func() {
				p.predictor.EndProfiling()
				profBuffer, err := p.predictor.ReadProfile()
				if err != nil {
					pp.Println(err)
					return
				}

				t, err := ctimer.New(profBuffer)
				if err != nil {
					panic(err)
				}
				t.Publish(ctx, tracer.FRAMEWORK_TRACE)

				p.predictor.DisableProfiling()
			}()
		}
	}

	if data == nil {
		return errors.New("input data nil")
	}
	input, ok := data.([]*gotensor.Dense)
	if !ok {
		return errors.New("input data is not slice of dense tensors")
	}

	err := p.predictor.SetInput(0, input[0].Float32s())
	if err != nil {
		return errors.Wrapf(err, "failed to set input")
	}

	err = p.predictor.Predict(ctx)
	if err != nil {
		return errors.Wrapf(err, "failed to perform Predict")
	}

	return nil
}

// ReadPredictedFeatures ...
func (p *ImageObjectDetectionPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	boxes, err := p.predictor.ReadOutputData(ctx, p.boxesLayerIndex)
	if err != nil {
		return nil, err
	}

	probabilities, err := p.predictor.ReadOutputData(ctx, p.probabilitiesLayerIndex)
	if err != nil {
		return nil, err
	}

	classes, err := p.predictor.ReadOutputData(ctx, p.classesLayerIndex)
	if err != nil {
		return nil, err
	}

	labels, err := p.GetLabels()
	if err != nil {
		return nil, errors.Wrap(err, "cannot get the labels")
	}

	return p.CreateBoundingBoxFeatures(ctx, probabilities, classes, boxes, labels)
}

func (p ImageObjectDetectionPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageObjectDetectionModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := caffe.FrameworkManifest
		agent.AddPredictor(framework, &ImageObjectDetectionPredictor{
			ImagePredictor: &ImagePredictor{
				ImagePredictor: common.ImagePredictor{
					Base: common.Base{
						Framework: framework,
					},
				},
			},
		})
	})
}
