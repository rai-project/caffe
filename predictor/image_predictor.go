package predictor

import (
	"context"

	// opentracing "github.com/opentracing/opentracing-go"

	opentracing "github.com/opentracing/opentracing-go"
	olog "github.com/opentracing/opentracing-go/log"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/options"
	common "github.com/rai-project/dlframework/framework/predictor"
	"github.com/rai-project/downloadmanager"
	gocaffe "github.com/rai-project/go-caffe"
	"github.com/rai-project/tracer"
)

// ImagePredictor ...
type ImagePredictor struct {
	common.ImagePredictor
	predictor *gocaffe.Predictor
}

// Close ...
func (p *ImagePredictor) Close() error {
	if p.predictor != nil {
		p.predictor.Close()
	}
	return nil
}

// Load ...
func (p *ImagePredictor) Load(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) (*ImagePredictor, error) {
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
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if ip.Options.DisableFrameworkAutoTuning() {
		log.Info("caffe does not have auto tuning")
	}

	if err = ip.download(ctx); err != nil {
		return nil, err
	}

	if err = ip.loadPredictor(ctx); err != nil {
		return nil, err
	}

	return ip, nil
}

func (p *ImagePredictor) loadPredictor(ctx context.Context) error {
	if ctx != nil {
		span, _ := tracer.StartSpanFromContext(ctx, tracer.APPLICATION_TRACE, "load_predictor")
		defer span.Finish()
	}

	opts, err := p.GetPredictionOptions()
	if err != nil {
		return err
	}

	pred, err := gocaffe.New(
		ctx,
		options.WithOptions(opts),
		options.Graph([]byte(p.GetGraphPath())),
		options.Weights([]byte(p.GetWeightsPath())),
	)
	if err != nil {
		return err
	}

	p.predictor = pred

	return nil
}

// Download ...
func (p *ImagePredictor) Download(ctx context.Context, model dlframework.ModelManifest, opts ...options.Option) error {
	framework, err := model.ResolveFramework()
	if err != nil {
		return err
	}

	workDir, err := model.WorkDir()
	if err != nil {
		return err
	}

	ip := &ImagePredictor{
		ImagePredictor: common.ImagePredictor{
			Base: common.Base{
				Framework: framework,
				Model:     model,
				WorkDir:   workDir,
				Options:   options.New(opts...),
			},
		},
	}

	if err = ip.download(ctx); err != nil {
		return err
	}

	return nil
}

func (p *ImagePredictor) download(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx,
		tracer.APPLICATION_TRACE,
		"download",
		opentracing.Tags{
			"graph_url":           p.GetGraphUrl(),
			"target_graph_file":   p.GetGraphPath(),
			"weights_url":         p.GetWeightsUrl(),
			"target_weights_file": p.GetWeightsPath(),
			"feature_url":         p.GetFeaturesUrl(),
			"target_feature_file": p.GetFeaturesPath(),
		},
	)
	defer span.Finish()

	model := p.Model

	if model.Model.IsArchive {
		baseURL := model.Model.BaseUrl
		span.LogFields(
			olog.String("event", "download model archive"),
		)
		if _, err := downloadmanager.DownloadInto(baseURL, p.WorkDir, downloadmanager.Context(ctx)); err != nil {
			return errors.Wrapf(err, "failed to download model archive from %v", model.Model.BaseUrl)
		}
	} else {
		span.LogFields(
			olog.String("event", "download model graph"),
		)

		_, err := downloadmanager.DownloadFile(
			p.GetGraphUrl(),
			p.GetGraphPath(),
			downloadmanager.MD5Sum(p.GetGraphChecksum()),
		)
		if err != nil {
			return err
		}

		span.LogFields(
			olog.String("event", "download weigths graph"),
		)

		_, err = downloadmanager.DownloadFile(
			p.GetWeightsUrl(),
			p.GetWeightsPath(),
			downloadmanager.MD5Sum(p.GetWeightsChecksum()),
		)
		if err != nil {
			return err
		}
	}

	if p.GetFeaturesUrl() != "" {
		span.LogFields(
			olog.String("event", "download features"),
		)
		_, err := downloadmanager.DownloadFile(
			p.GetFeaturesUrl(),
			p.GetFeaturesPath(),
			downloadmanager.MD5Sum(p.GetFeaturesChecksum()),
		)
		if err != nil {
			return err
		}
	}

	return nil
}
