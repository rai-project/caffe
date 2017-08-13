package main

import (
	"fmt"
	"os"

	"github.com/rai-project/caffe"
	_ "github.com/rai-project/caffe/predict"
	"github.com/rai-project/dlframework/framework/cmd"
)

func main() {
	cmd.Init()

	rootCmd, err := cmd.NewRootCommand(caffe.FrameworkManifest)
	if err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(-1)
	}
}
