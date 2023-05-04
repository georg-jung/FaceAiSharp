// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Numerics;
using SimpleSimd;

namespace FaceAiSharp.Simd;

internal static class ElementwiseGreater
{
    public static void Greater(this ReadOnlySpan<float> left, float right, Span<int> result)
    {
        SimdOps<float>.Concat(left, right, default(Greater_VSelector), default(Greater_Selector), result);
    }

    public static void Greater(this ReadOnlySpan<float> left, ReadOnlySpan<float> right, Span<int> result)
    {
        SimdOps<float>.Concat(left, right, default(Greater_VSelector), default(Greater_Selector), result);
    }

    private struct Greater_VSelector : IFunc<Vector<float>, Vector<float>, Vector<int>>
    {
        public Vector<int> Invoke(Vector<float> left, Vector<float> right)
        {
            return Vector.GreaterThan(left, right);
        }
    }

    private struct Greater_Selector : IFunc<float, float, int>
    {
        public int Invoke(float left, float right)
        {
            return left > right ? 1 : 0;
        }
    }
}
