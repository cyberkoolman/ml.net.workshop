public class TransactionFraudPrediction : IModelEntity
{
  public float Label;

  /// <summary>
  /// The non-negative, unbounded score that was calculated by the anomaly detection model.
  /// Fraudulent transactions (Anomalies) will have higher scores than normal transactions
  /// </summary>
  public float Score;

  /// <summary>
  /// The predicted label, based on the score. A value of true indicates an anomaly.
  /// </summary>
  public bool PredictedLabel;

  public void PrintToConsole()
  {
    Console.WriteLine($"Predicted Label: {PredictedLabel}  (Score: {Score})");
  }
}