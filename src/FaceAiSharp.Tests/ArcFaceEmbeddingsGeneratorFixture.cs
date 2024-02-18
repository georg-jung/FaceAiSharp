// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Microsoft.ML.OnnxRuntime;

namespace FaceAiSharp.Tests;

public sealed class ArcFaceEmbeddingsGeneratorFixture : IDisposable
{
    public ArcFaceEmbeddingsGeneratorFixture()
    {
        Embedder = CreateArcFaceEmbedder();
    }

    public ArcFaceEmbeddingsGenerator Embedder { get; }

    public void Dispose() => Embedder.Dispose();

    private static ArcFaceEmbeddingsGenerator CreateArcFaceEmbedder(SessionOptions? sessionOptions = null)
    {
        var modelPath = Path.Combine(GetExeDir(), "onnx", "arcfaceresnet100-11-int8.onnx");
        var opt = new ArcFaceEmbeddingsGeneratorOptions() { ModelPath = modelPath };
        return new ArcFaceEmbeddingsGenerator(opt, sessionOptions);
    }

    private static string GetExeDir() => Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!;
}
