package neural_plswork.network;

import neural_plswork.connection.penalize.Penalty;
import neural_plswork.core.NetworkValue;
import neural_plswork.layers.basic.InputLayer;
import neural_plswork.math.Vector;
import neural_plswork.neuron.evaluation.Differentiable;
import neural_plswork.neuron.evaluation.Evaluation;
import neural_plswork.unit.Unit;
import neural_plswork.unit.ffUnits.HiddenUnit;
import neural_plswork.unit.ffUnits.OutputUnit;

public class Network {
    private final InputLayer input;
    private final HiddenUnit[] hidden;
    private final OutputUnit output;
    
    private Differentiable evaluator;
    private Evaluation reporter;
    

    protected Network(InputLayer input, HiddenUnit[] hidden, OutputUnit output, Differentiable evaluator, Evaluation reporter, Penalty penalty) {
        this.input = input;
        this.hidden = hidden;
        this.output = output;

        this.evaluator = evaluator;
        this.reporter = reporter;
    }

    public double[] test(double... inputs) {
        Vector<NetworkValue> vector = NetworkValue.arrToVector(inputs);
        input.setInputs(vector, 0);
        for(Unit u: hidden) u.forwardPass(0);
        output.forwardPass(0);

        return NetworkValue.vectorToArr(output.getOutputs(0));
    }
}
