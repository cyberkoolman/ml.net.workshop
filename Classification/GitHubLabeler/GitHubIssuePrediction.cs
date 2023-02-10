﻿#pragma warning disable 649 // We don't care about unused fields here, because they are mapped with the input file.
using Microsoft.ML.Data;

public class GitHubIssuePrediction
{
    [ColumnName("PredictedLabel")]
    public string Area;

    public float[] Score;
}
