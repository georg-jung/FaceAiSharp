// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using SimpleSimd;

namespace FaceAiSharp.Simd;

internal static class ElementwiseMin
{
    public static void Min(this ReadOnlySpan<float> left, float right, Span<float> result)
    {
        SimdOps<float>.Concat(left, right, default(Min_VSelector), default(Min_Selector), result);
    }

    public static void Min(this ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<float> result)
    {
        SimdOps<float>.Concat(left, right, default(Min_VSelector), default(Min_Selector), result);
    }

    private struct Min_VSelector : IFunc<Vector<float>, Vector<float>, Vector<float>>
    {
        public Vector<float> Invoke(Vector<float> left, Vector<float> right)
        {
            return Vector.Min(left, right);
        }
    }

    private struct Min_Selector : IFunc<float, float, float>
    {
        public float Invoke(float left, float right)
        {
            return MathF.Min(left, right);
        }
    }
}
