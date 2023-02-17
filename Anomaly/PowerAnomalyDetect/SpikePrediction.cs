using Microsoft.ML.Data;

class SpikePrediction
{
    [VectorType(3)]
    public double[] Prediction { get; set; }
}