// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using BenchmarkDotNet.Attributes;
using FaceAiSharp.Simd;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using SimpleSimd;

namespace Benchmarks.NonMaxSupression;

[MemoryDiagnoser]
public class Benchmarks
{
    private readonly string _folderPath = $@"{Path.GetDirectoryName(typeof(Benchmarks).Assembly.Location)}/TestData/NMS/";
    private readonly NDArray _arr1;
    private readonly NDArray _arr2;
    private readonly NDArray _arr3;

    public Benchmarks()
    {
        // depending on input data size the vectorized version provides a speedup of 10-50x
        _arr1 = np.load($@"{_folderPath}/crowd.npy");
        _arr2 = np.load($@"{_folderPath}/group.npy");
        _arr3 = np.load($@"{_folderPath}/portrait.npy");
    }

    [Benchmark]
    public List<int> FromScrfdPyCrowd() => FromScrfdPy(_arr1, 0.4f);

    [Benchmark]
    public List<int> FromScrfdPyGroup() => FromScrfdPy(_arr2, 0.4f);

    [Benchmark]
    public List<int> FromScrfdPyPortrait() => FromScrfdPy(_arr3, 0.4f);

    [Benchmark]
    public List<int> VectorziedCrowd() => Vectorized(_arr1, 0.4f);

    [Benchmark]
    public List<int> VectorziedGroup() => Vectorized(_arr2, 0.4f);

    [Benchmark]
    public List<int> VectorziedPortrait() => Vectorized(_arr3, 0.4f);

    /// <summary>
    /// In real numpy this could be eg. np.where(scores&lt;=thresh) or np.asarray(condition).nonzero().
    /// </summary>
    /// <param name="input">The indices of this array's elements should be returned.</param>
    /// <param name="threshold">The threshold value. Exclusive.</param>
    /// <returns>An NDArray contianing the indices.</returns>
    private static NDArray IndicesOfElementsBelow(NDArray input, float threshold)
    {
        var zeroIfAbove = np.sign(input - threshold) - 1;
        var ret = np.nonzero(zeroIfAbove);
        return ret[0];
    }

    /// <summary>
    /// Filter out duplicate detections (multiple boxes describing roughly the same area) using non max suppression.
    /// </summary>
    /// <param name="dets">All detections with their scores.</param>
    /// <param name="thresh">Non max suppression threshold.</param>
    /// <returns>Which detections to keep.</returns>
    private static List<int> FromScrfdPy(NDArray dets, float thresh)
    {
        var x1 = dets[":, 0"];
        var y1 = dets[":, 1"];
        var x2 = dets[":, 2"];
        var y2 = dets[":, 3"];
        var scores = dets[":, 4"];
        var areas = (x2 - x1 + 1) * (y2 - y1 + 1);

        // the clone should not be needed but NumSharp returns false values otherwise
        var order = scores.Clone().argsort<float>()["::-1"];

        List<int> keep = new();
        while (order.size > 0)
        {
            var i = order[0];
            keep.Add(i);

            if (order.size == 1)
            {
                break;
            }

            var xx1 = np.maximum(x1[i], x1[order["1:"]]);
            var yy1 = np.maximum(y1[i], y1[order["1:"]]);
            var xx2 = np.minimum(x2[i], x2[order["1:"]]);
            var yy2 = np.minimum(y2[i], y2[order["1:"]]);

            var w = np.maximum(0.0, xx2 - xx1 + 1);
            var h = np.maximum(0.0, yy2 - yy1 + 1);
            var inter = w * h;
            var ovr = inter / (areas[i] + areas[order["1:"]] - inter);

            // this is <= in python but < here
            var inds = IndicesOfElementsBelow(ovr, thresh).Clone();

            // NumSharp has a bug that throws an OutOfRangeException if we use the same NDArray on the
            // left and right side of a variable assignment.
            order = np.array(order, true)[inds + 1];
        }

        return keep;
    }

    /// <summary>
    /// Filter out duplicate detections (multiple boxes describing roughly the same area) using non max suppression.
    /// </summary>
    /// <param name="dets">All detections with their scores.</param>
    /// <param name="thresh">Non max suppression threshold.</param>
    /// <returns>Which detections to keep.</returns>
    private static List<int> Vectorized(NDArray dets, float thresh)
    {
        var x1 = ((IArraySlice)dets[":, 0"].Data<float>()).AsSpan<float>();
        var y1 = ((IArraySlice)dets[":, 1"].Data<float>()).AsSpan<float>();
        var x2 = ((IArraySlice)dets[":, 2"].Data<float>()).AsSpan<float>();
        var y2 = ((IArraySlice)dets[":, 3"].Data<float>()).AsSpan<float>();

        int len = x1.Length;
        var p = SimdOps<float>.Subtract(x2, x1);
        var q = SimdOps<float>.Subtract(y2, y1);
        SimdOps<float>.Add(p, 1f, p);
        SimdOps<float>.Add(q, 1f, q);
        var areasArr = SimdOps<float>.Multiply(p, q);
        var areas = areasArr.AsSpan();

        var discard = new int[len];
        var pivot_discard = new int[len];
        var xx1 = new float[len];
        var xx2 = new float[len];
        var yy1 = new float[len];
        var yy2 = new float[len];
        var keep = new List<int>(len);
        for (var i = 0; i < len; i++)
        {
            if (discard[i] != 0)
            {
                continue;
            }

            keep.Add(i);

            var (i_x1, i_x2, i_y1, i_y2, i_area) = (x1[i], x2[i], y1[i], y2[i], areas[i]);
            ElementwiseMax.Max(x1, i_x1, xx1);
            ElementwiseMax.Max(y1, i_y1, yy1);
            ElementwiseMin.Min(x2, i_x2, xx2);
            ElementwiseMin.Min(y2, i_y2, yy2);

            SimdOps<float>.Subtract(xx2, xx1, xx1);
            SimdOps<float>.Add(xx1, 1f, xx1);
            ElementwiseMax.Max(xx1, 0f, xx1); // xx1 = w

            SimdOps<float>.Subtract(yy2, yy1, yy1);
            SimdOps<float>.Add(yy1, 1f, yy1);
            ElementwiseMax.Max(yy1, 0f, yy1); // yy1 = h

            SimdOps<float>.Multiply(xx1, yy1, xx1); // xx1 = inter
            SimdOps<float>.Add(areas, i_area, yy1);
            SimdOps<float>.Subtract(yy1, xx1, yy1); // yy1 = (areas[i] + areas[order["1:"]] - inter)
            SimdOps<float>.Divide(xx1, yy1, xx1); // xx1 = ovr = inter / (areas[i] + areas[order["1:"]] - inter)

            ElementwiseGreater.Greater(xx1, thresh, pivot_discard);
            SimdOps<int>.Add(discard, pivot_discard, discard);
        }

        return keep;
    }
}
