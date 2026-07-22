// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Exporters.Json;
using BenchmarkDotNet.Running;

namespace Benchmarks;

public class Program
{
    public static void Main(string[] args)
    {
        var cfg = ManualConfig
            .Create(DefaultConfig.Instance)

            // required for OnnxRuntime
            .WithOptions(ConfigOptions.DisableOptimizationsValidator)
            .AddDiagnoser(MemoryDiagnoser.Default)

            // the full-compressed json report is the format benchmark-action/github-action-benchmark parses
            .AddExporter(JsonExporter.FullCompressed);

        // Select benchmarks via command line, e.g.
        //   dotnet run -c Release -- --anyCategories regression
        //   dotnet run -c Release -- --filter '*FaceDetection*'
        //   dotnet run -c Release -- --list flat
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, cfg);
    }
}
