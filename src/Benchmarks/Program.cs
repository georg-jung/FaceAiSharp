// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace Benchmarks;

public class Program
{
    public static void Main(string[] args)
    {
        // required for OnnxRuntime
        var opt = ManualConfig
            .Create(DefaultConfig.Instance)
            .WithOptions(ConfigOptions.DisableOptimizationsValidator);

        // var summary = BenchmarkRunner.Run<ImageSharp>();
        // var summary = BenchmarkRunner.Run<FaceOnnxVsImageSharpAlignment>();
        // var summary = BenchmarkRunner.Run<CropFirstVsResizeFirst>(args: args);
        // var summary = BenchmarkRunner.Run<Scrfd>(opt, args);
        // var summary = BenchmarkRunner.Run<ImageToTensor.Benchmarks>(opt, args);
        // var summary = BenchmarkRunner.Run<NonMaxSupression.Benchmarks>(opt, args);
        var summary = BenchmarkRunner.Run<ImageCloneVsMutate>(opt, args);
    }
}
