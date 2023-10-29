// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

namespace FaceAiSharp;

public static class Metrics
{
    /// <summary>
    /// Calculate the "area under the ROC curve" value for the given estimations. AUC is a quality metric
    /// for a classifier. Higher is better. Below 0.5 is useless for a binary classifier.
    /// </summary>
    /// <param name="estimations">
    ///     A list of tuples of confidence values our model calculated with their the corresponding
    ///     ground-truth values.
    /// </param>
    /// <returns>The "area under the ROC curve" value for the given estimations.</returns>
    public static float Auc(IReadOnlyList<(float Confidence, bool IsMatch)> estimations)
    {
        var auc = 0.0;
        var height = 0.0;
        var idx = 0;
        var curPos = 0;
        double p = estimations.Count(x => x.IsMatch);
        double n = estimations.Count - p;
        foreach (var (c, m) in estimations)
        {
            idx++;

            if (m)
            {
                curPos++;
                height += 1 / p;
            }
            else
            {
                auc += height * /* fpr */ (1 - ((idx - curPos) / n));
                height = 0;
            }
        }

        return (float)auc;
    }

    /// <summary>
    /// Generates the points needed to draw a ROC curve for the classifier that generated the estimations you pass as arguments.
    /// </summary>
    /// <param name="estimations">
    ///     A list of tuples of confidence values our model calculated with their the corresponding
    ///     ground-truth values.
    /// </param>
    /// <returns>Points to draw in a ROC curve.</returns>
    public static IEnumerable<(float X_FPR, float Y_TPR, float Threshold)> RocPoints(IReadOnlyList<(float Confidence, bool IsMatch)> estimations)
    {
        var idx = 0;
        var curPos = 0;
        float p = estimations.Count(x => x.IsMatch);
        float n = estimations.Count - p;
        foreach (var (c, m) in estimations)
        {
            idx++;

            if (m)
            {
                curPos++;
            }

            yield return ((idx - curPos) / n, curPos / p, c);
        }
    }

    /// <summary>
    /// Finds an optimal threshold for a binary classifier. An optimal threshold is defined as the threshold
    /// that leads to the highest possible accuracy.
    /// </summary>
    /// <param name="estimations">
    ///     A list of tuples of confidence values our model calculated with their the corresponding
    ///     ground-truth values.
    /// </param>
    /// <returns>A threshold that leads to the highest possible accuracy.</returns>
    public static float FindThreshold(IReadOnlyList<(float Confidence, bool IsMatch)> estimations)
    {
        var idx = 0;
        var tp = 0;
        float p = estimations.Count(x => x.IsMatch);
        float n = estimations.Count - p;
        float pivot = 0.0f;
        var acc = 0.0;
        foreach (var (c, m) in estimations)
        {
            idx++;

            if (m)
            {
                tp++;
            }

            var fp = idx - tp;
            var tn = n - fp;
            var i_acc = (tp + tn) / (p + n);
            if (i_acc > acc)
            {
                acc = i_acc;
                pivot = c;
            }
        }

        return pivot;
    }
}
