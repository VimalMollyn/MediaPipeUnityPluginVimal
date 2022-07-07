using UnityEngine;

namespace Mediapipe.Unity.Tutorial
{
  public class HelloWorld : MonoBehaviour
  {
    private void Start()
    {
      var configText = @"
input_stream: ""in""
output_stream: ""out""
node {
  calculator: ""PassThroughCalculator""
  input_stream: ""in""
  output_stream: ""out1""
}
node {
  calculator: ""PassThroughCalculator""
  input_stream: ""out1""
  output_stream: ""out""
}
";
      var graph = new CalculatorGraph(configText);
      var poller = graph.AddOutputStreamPoller<string>("out").Value();
      graph.StartRun().AssertOk();

      for (var i = 0; i < 10; i++)
      {
        var input = new StringPacket("Hello World!", new Timestamp(i));
        graph.AddPacketToInputStream("in", input).AssertOk();
      }

      graph.CloseInputStream("in").AssertOk();

      var output = new StringPacket();
      while (poller.Next(output))
      {
        Debug.Log(output.Get());
      }

      graph.WaitUntilDone().AssertOk();
      graph.Dispose();

      Debug.Log("Done");
    }
  }
}
