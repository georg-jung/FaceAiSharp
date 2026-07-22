// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Benchmarks.Regression;

/// <summary>
/// Resolves models and test images that are copied to the build output.
/// </summary>
internal static class BenchmarkData
{
    public static string ModelPath(string modelFileName) => Path.Combine(AppContext.BaseDirectory, "onnx", modelFileName);

    public static string TestDataPath(params string[] parts) => Path.Combine([AppContext.BaseDirectory, "TestData", .. parts]);

    public static Image<Rgb24> LoadImage(string jpgFileName) => Image.Load<Rgb24>(TestDataPath("jpgs", jpgFileName));
}
