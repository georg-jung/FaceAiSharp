// Copyright (c) Georg Jung. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using FaceAiSharp.Extensions;
using Humanizer;
using LiteDB;

namespace FaceAiSharp.Validation;

internal class CalculateAllDistances
{
    private readonly FileInfo _db;
    private readonly string _dbEmbeddingCollectionName;
    private readonly float _threshold;

    public CalculateAllDistances(FileInfo db, string dbEmbeddingCollectionName, float threshold)
    {
        _db = db;
        _dbEmbeddingCollectionName = dbEmbeddingCollectionName;
        _threshold = threshold;
    }

    public void Invoke()
    {
        var sw = Stopwatch.StartNew();
        using var db = new LiteDatabase(_db.FullName);
        var dbEmb = db.GetCollection<Embedding>(_dbEmbeddingCollectionName);

        var embs = dbEmb.FindAll().ToList();

        double avgCosDist = 0;
        double avgEuclDist = 0;
        double avgDotProd = 0;
        double avgCosDistTrue = 0;
        double avgEuclDistTrue = 0;
        double avgDotProdTrue = 0;
        double avgCosDistFalse = 0;
        double avgEuclDistFalse = 0;
        double avgDotProdFalse = 0;
        int cnt = 0;
        int trueCnt = 0;
        int totalCnt = GaussianSum(embs.Count - 1);

        var tp = 0;
        var tn = 0;
        var fp = 0;
        var fn = 0;

        void ReportProgress()
        {
            Console.WriteLine($"Calculated {cnt} pairs. {(double)cnt / totalCnt:P2}");
            Console.WriteLine($"Had {trueCnt} pairs belonging to the same person.");
            Console.WriteLine($"Avergage cosine distance:                   {avgCosDist / cnt}");
            Console.WriteLine($"Avergage euclidean distance:                {avgEuclDist / cnt}");
            Console.WriteLine($"Avergage dot product:                       {avgDotProd / cnt}");
            Console.WriteLine($"Avergage cosine distance [same person]:     {avgCosDistTrue / trueCnt}");
            Console.WriteLine($"Avergage euclidean distance [same person]:  {avgEuclDistTrue / trueCnt}");
            Console.WriteLine($"Avergage dot product [same person]:         {avgDotProdTrue / trueCnt}");
            Console.WriteLine($"Avergage cosine distance [diff. person]:    {avgCosDistFalse / (cnt - trueCnt)}");
            Console.WriteLine($"Avergage euclidean distance [diff. person]: {avgEuclDistFalse / (cnt - trueCnt)}");
            Console.WriteLine($"Avergage dot product [diff. person]:        {avgDotProdFalse / (cnt - trueCnt)}");
            Console.WriteLine($"P: {tp + fn,8:D} TP: {tp,8:D} FP: {fp,8:D}");
            Console.WriteLine($"N: {tn + fp,8:D} TN: {tn,8:D} FN: {fn,8:D}");
            Console.WriteLine($"Accuracy:  {(double)(tp + tn) / (tp + tn + fp + fn):P2}");
            Console.WriteLine($"Precision: {(double)tp / (tp + fp):P2}");
            Console.WriteLine($"Recall:    {(double)tp / (tp + fn):P2}");
            Console.WriteLine($"F1 score:  {(double)tp * 2 / ((tp * 2) + fp + fn):N4}");
            Console.WriteLine();
        }

        for (var x = 0; x < embs.Count; x++)
        {
            var embX = embs[x];
            for (var y = x + 1; y < embs.Count; y++)
            {
                var embY = embs[y];
                var cosDist = embX.Embeddings.CosineDistance(embY.Embeddings);
                var euclDist = embX.Embeddings.EuclideanDistance(embY.Embeddings);
                var dotProd = embX.Embeddings.Dot(embY.Embeddings);
                var sameIdnt = embX.Identity.Equals(embY.Identity, StringComparison.Ordinal);
                avgCosDist += cosDist;
                avgEuclDist += euclDist;
                avgDotProd += dotProd;
                if (sameIdnt)
                {
                    avgCosDistTrue += cosDist;
                    avgEuclDistTrue += euclDist;
                    avgDotProdTrue += dotProd;
                    trueCnt++;
                }
                else
                {
                    avgCosDistFalse += cosDist;
                    avgEuclDistFalse += euclDist;
                    avgDotProdFalse += dotProd;
                }

#pragma warning disable SA1503 // Braces should not be omitted
                if (dotProd > _threshold && sameIdnt) tp++;
                if (dotProd > _threshold && !sameIdnt) fp++;
                if (dotProd <= _threshold && sameIdnt) fn++;
                if (dotProd <= _threshold && !sameIdnt) tn++;
#pragma warning restore SA1503 // Braces should not be omitted

                cnt++;

                // takes around 5s on my notebook
                if (cnt % 5000000 == 0)
                {
                    ReportProgress();
                }
            }
        }

        Console.WriteLine($"--- FINISHED! Final Stats {_threshold} ---");
        ReportProgress();
        Console.WriteLine($"Took {sw.Elapsed.Humanize(4)}");
    }

    private static int GaussianSum(int n) => n * (n + 1) / 2;
}
