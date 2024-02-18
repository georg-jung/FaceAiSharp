// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Reflection;
using Microsoft.ML.OnnxRuntime;

namespace FaceAiSharp.Tests;

public sealed class OpenVinoOpenClosedEye0001Fixture : IDisposable
{
    public OpenVinoOpenClosedEye0001Fixture()
    {
        OpenClosed = CreateOpenClosedEye0001();
    }

    public OpenVinoOpenClosedEye0001 OpenClosed { get; }

    public void Dispose() => OpenClosed.Dispose();

    private static OpenVinoOpenClosedEye0001 CreateOpenClosedEye0001(SessionOptions? sessionOptions = null)
    {
        var modelPath = Path.Combine(GetExeDir(), "onnx", "open_closed_eye.onnx");
        var opt = new OpenVinoOpenClosedEye0001Options() { ModelPath = modelPath };
        return new OpenVinoOpenClosedEye0001(opt, sessionOptions);
    }

    private static string GetExeDir() => Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!;
}
