// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using FaceAiSharp.Simd;
using NumSharp;

namespace FaceAiSharp.Tests;

public class NonMaxSupressionCorrectness
{
    private readonly string _folderPath = $@"{Path.GetDirectoryName(typeof(NonMaxSupressionCorrectness).Assembly.Location)}/TestData/NMS/";
    private readonly NDArray _arr1;
    private readonly NDArray _arr2;
    private readonly NDArray _arr3;

    public NonMaxSupressionCorrectness()
    {
        // depending on input data size the vectorized version provides a speedup of 10-50x
        _arr1 = np.load($@"{_folderPath}/crowd.npy");
        _arr2 = np.load($@"{_folderPath}/group.npy");
        _arr3 = np.load($@"{_folderPath}/portrait.npy");
    }

    [Fact]
    public void CurrentImplementationMatchesScrfdPy()
    {
        var threshs = new[] { 0.4f, 0.5f };
        foreach (var thresh in threshs)
        {
            TestInput(_arr1, thresh);
            TestInput(_arr2, thresh);
            TestInput(_arr3, thresh);
        }
    }

    private static void TestInput(NDArray input, float thresh)
    {
        var res1 = ScrfdDetector.NonMaxSupression(input, thresh);
        var res2 = FromScrfdPy(input, thresh);
        Assert.Equal(res1, res2);
    }

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
}
