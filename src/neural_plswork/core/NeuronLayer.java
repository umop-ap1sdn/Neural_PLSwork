package neural_plswork.core;

import java.util.LinkedList;

import neural_plswork.activations.ActivationFunction;
import neural_plswork.math.Vector;

@SuppressWarnings("unused")
public abstract class NeuronLayer {
    
    private LinkedList<Vector<NetworkValue>> unactivated;
    private LinkedList<Vector<NetworkValue>> activated;
    private LinkedList<Vector<NetworkValue>> derivative;
    private LinkedList<Vector<NetworkValue>> error;

    private ActivationFunction activation;

    private int layerSize;
    private int historySize;
    private boolean bias;
}
