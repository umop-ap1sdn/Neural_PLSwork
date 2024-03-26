package neural_plswork.core;

import neural_plswork.initialize.Initializer;
import neural_plswork.layers.basic.InputLayer;
import neural_plswork.layers.basic.OutputLayer;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;

public class ConnectionLayer {
    
    Matrix<NetworkValue> primaryLayer;
    Vector<NetworkValue> biasVector;

    private NeuronLayer srcLayer;
    private NeuronLayer destLayer;

    private Initializer initializer;

    public ConnectionLayer(NeuronLayer srcLayer, NeuronLayer destLayer, Initializer initializer) throws InvalidNetworkConstructionException {
        if(srcLayer instanceof OutputLayer) 
            throw new InvalidNetworkConstructionException("Cannot use Output Layer as Source Layer");
        if(destLayer instanceof InputLayer)
            throw new InvalidNetworkConstructionException("Cannot use Input Layer as Dest Layer");
        
        this.srcLayer = srcLayer;
        this.destLayer = destLayer;
        this.initializer = initializer;

        this.initialize();

    }

    private void initialize() {
        int rows = destLayer.size();
        int cols = srcLayer.size();

        NetworkValue[][] matrix = new NetworkValue[rows][cols];

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                matrix[i][j] = new NetworkValue(initializer.getNextWeight());
            }
        }

        this.primaryLayer = new Matrix<>(matrix);

        NetworkValue[] vector = new NetworkValue[rows];

        for(int i = 0; i < rows; i++) {
            if(srcLayer.getBias()) vector[i] = new NetworkValue(initializer.getNextWeight());
            else vector[i] = new NetworkValue();
        }

        this.biasVector = new Vector<>(vector);
    }

    public Vector<NetworkValue> forwardPass(Vector<NetworkValue> input) {
        Matrix<NetworkValue> multiplied = primaryLayer.multiply(input);
        Matrix<NetworkValue> added = multiplied.add(biasVector);
        return added.getAsVector();
    }

    public Vector<NetworkValue> forwardPass() {
        return forwardPass(srcLayer.getRecentValues());
    }
}
