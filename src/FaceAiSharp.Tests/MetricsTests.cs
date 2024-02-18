// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using Shouldly;

namespace FaceAiSharp.Tests;

public class MetricsTests
{
    [Fact]
    public void Auc_ShouldReturnCorrectValue()
    {
        Metrics.Auc(new List<(float Confidence, bool IsMatch)>
        {
            (0.9f, true),
            (0.8f, true),
            (0.75f, true),
            (0.7f, true),
            (0.3f, false),
            (0.2f, false),
            (0.1f, false),
        }).ShouldBeGreaterThan(0.5f);
    }

    [Fact]
    public void FindThreshold_ShouldReturnOptimalThreshold()
    {
        Metrics.FindThreshold(new List<(float Confidence, bool IsMatch)>
        {
            (0.9f, true),
            (0.1f, false),
        }).ShouldBeInRange(0.100001f, 0.9f);

        Metrics.FindThreshold(new List<(float Confidence, bool IsMatch)>
        {
            (0.9f, true),
            (0.9f, true),
            (0.9f, true),
            (0.8f, true),
            (0.7f, true),
            (0.1f, false),
        }).ShouldBeInRange(0.100001f, 0.7f);

        Metrics.FindThreshold(new List<(float Confidence, bool IsMatch)>
        {
            (0.9f, true),
            (0.9f, true),
            (0.9f, true),
            (0.8f, true),
            (0.7f, false),
            (0.5f, true),
            (0.6f, true),
            (0.1f, false),
        }).ShouldBeInRange(0.100001f, 0.6f);
    }
}
