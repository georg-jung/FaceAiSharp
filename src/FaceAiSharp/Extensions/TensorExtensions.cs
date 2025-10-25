// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FaceAiSharp.Extensions;

internal static class TensorExtensions
{
    public static T[] ToArray<T>(this NamedOnnxValue input)
    {
        var denseTensor = input.Value as DenseTensor<T> ?? input.AsTensor<T>().ToDenseTensor();
        return denseTensor.ToArray();
    }
}
