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

type SemanticSegmentationPredictor struct {
	*ImagePredictor
	masksLayerIndex int
	masks           interface{}
}

// New ...
func NewSemanticSegmentationPredictor(model dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {
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

	predictor := new(SemanticSegmentationPredictor)
	return predictor.Load(ctx, model, opts...)
}

func (self *SemanticSegmentationPredictor) Load(ctx context.Context, modelManifest dlframework.ModelManifest, opts ...options.Option) (common.Predictor, error) {

	pred, err := self.ImagePredictor.Load(ctx, modelManifest, opts...)
	if err != nil {
		return nil, err
	}
	p := &SemanticSegmentationPredictor{
		ImagePredictor: pred,
	}

	p.masks, err = p.GetOutputLayerIndex("masks_layer")
	if err != nil {
		return nil, errors.Wrap(err, "failed to get the masks layer index")
	}

	return p, nil
}

// Predict ...
func (p *SemanticSegmentationPredictor) Predict(ctx context.Context, data interface{}, opts ...options.Option) error {
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
func (p *SemanticSegmentationPredictor) ReadPredictedFeatures(ctx context.Context) ([]dlframework.Features, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "read_predicted_features")
	defer span.Finish()

	masks, err := p.predictor.ReadOutputData(ctx, p.masksLayerIndex)
	if err != nil {
		return nil, err
	}

	labels, err := p.GetLabels()
	if err != nil {
		return nil, errors.Wrap(err, "cannot get the labels")
	}

	return p.CreateSemanticSegmentFeatures(ctx, masks, labels)
}

func (p SemanticSegmentationPredictor) Modality() (dlframework.Modality, error) {
	return dlframework.ImageSemanticSegmentationModality, nil
}

func init() {
	config.AfterInit(func() {
		framework := caffe.FrameworkManifest
		agent.AddPredictor(framework, &SemanticSegmentationPredictor{
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
