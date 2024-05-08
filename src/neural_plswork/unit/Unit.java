package neural_plswork.unit;

import neural_plswork.connection.optimizer.OptimizationFunction;
import neural_plswork.connection.optimizer.Optimizer;
import neural_plswork.connection.penalize.Penalty;
import neural_plswork.connection.penalize.WeightPenalizer;
import neural_plswork.core.ConnectionLayer;
import neural_plswork.core.NetworkValue;
import neural_plswork.core.NeuronLayer;
import neural_plswork.math.Matrix;
import neural_plswork.math.Vector;
import neural_plswork.neuron.dropout.Dropout;
import neural_plswork.neuron.dropout.DropoutRegularizer;

public abstract class Unit {
    final protected NeuronLayer[] nLayers;
    final protected ConnectionLayer[] cLayers;

    final protected int batchSize;

    private double DEFAULT_L1 = 0.1;
    private double DEFAULT_L2 = 0.1;
    private double DEFAULT_P = 0.2;

    private Matrix<NetworkValue>[][] primaryGradients;
    private Vector<NetworkValue>[][] biasGradients;

    @SuppressWarnings("unchecked")
    public Unit(NeuronLayer[] nLayers, ConnectionLayer[] cLayers, int batchSize, int max_threads) {
        this.nLayers = nLayers;
        this.cLayers = cLayers;
        this.batchSize = batchSize;

        primaryGradients = new Matrix[cLayers.length][max_threads];
        biasGradients = new Vector[cLayers.length][max_threads];
    }

    public void clear(int thread) {
        for(NeuronLayer n: nLayers) n.clear(thread);
    }

    public void clear() {
        for(NeuronLayer n: nLayers) n.clear();
    }

    public void calculateGradients(double lr, boolean descending, int thread) {
        int index = 0;
        for(ConnectionLayer c: cLayers) {
            Matrix<NetworkValue>[] gradients = c.calculateGradients(lr, batchSize, descending, thread);
            primaryGradients[index][thread] = gradients[0];
            biasGradients[index][thread] = gradients[1].getAsVector();
            index++;
        }
    }

    public void adjustWeights(int thread) {
        int index = 0;
        for(ConnectionLayer c: cLayers) {
            c.adjustWeights(primaryGradients[index][thread], biasGradients[index][thread]);
            index++;
        }
    } 

    public final boolean validityCheck(int historySize, int MAX_THREADS) {
        for(NeuronLayer n: nLayers) {
            if(n.HISTORY_SIZE() != historySize || n.MAX_THREADS() != MAX_THREADS) return false;
        }

        return true;
    }

    public void setDefaultLambdas(double l1, double l2) {
        this.DEFAULT_L1 = l1;
        this.DEFAULT_L2 = l2;
    }

    public void setPenalty(Penalty[] penalty) {
        for(int i = 0; i < cLayers.length; i++) {
            if(penalty[i % penalty.length] == null) continue;
            cLayers[i].setPenalty(penalty[i % penalty.length]);
        }
    }

    public void setPenalty(WeightPenalizer[] penalty) {
        for(int i = 0; i < cLayers.length; i++) {
            if(penalty[i % penalty.length] == null) continue;
            cLayers[i].setPenalty(Penalty.getPenalty(penalty[i % penalty.length], DEFAULT_L1, DEFAULT_L2));
        }
    }

    public void setOptimizer(OptimizationFunction[] optimizer) {
        for(int i = 0; i < cLayers.length; i++) {
            if(optimizer[i % optimizer.length] == null) continue;
            cLayers[i].setOptimizer(optimizer[i % optimizer.length]);
        }
    }

    public void setOptimizer(Optimizer[] optimizer) {
        for(int i = 0; i < cLayers.length; i++) {
            if(optimizer[i % optimizer.length] == null) continue;
            cLayers[i].setOptimizer(OptimizationFunction.getFunction(optimizer[i % optimizer.length]));
        }
    }

    public void setDropout(Dropout[] dropout) {
        for(int i = 0; i < nLayers.length; i++) {
            if(dropout[i % dropout.length] == null) continue;
            nLayers[i].setDropout(dropout[i % dropout.length]);
        }
    }

    public void setDropout(DropoutRegularizer[] dropout) {
        for(int i = 0; i < nLayers.length; i++) {
            if(dropout[i % dropout.length] == null) continue;
            nLayers[i].setDropout(Dropout.getDropout(dropout[i % dropout.length], DEFAULT_P, batchSize));
        }
    }

    public void setDropoutEnable(boolean dropout_enabled) {
        for(NeuronLayer n: nLayers) {
            n.setDropoutEnable(dropout_enabled);
        }
    }

    public void setDefaultP(double p) {
        this.DEFAULT_P = p;
    }

    public double getPenaltySum() {
        double sum = 0.0;
        for(ConnectionLayer c: cLayers) {
            sum += c.getPenaltySum();
        }

        return sum;
    }

    public String unitToString(int resolution) {
        StringBuilder sb = new StringBuilder();
        sb.append(cLayers.length);
        sb.append("\n");
        for(ConnectionLayer c: cLayers) {
            sb.append(c.layerToString(resolution));
        }
        return sb.toString();
    }

    public String unitConfigString() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.getClass());
        sb.append("\n");
        if(nLayers.length == 0) return sb.append("\n\n").toString();
        
        sb.append(nLayers[0].size());
        for(int i = 1; i < nLayers.length; i++) {
            sb.append(',');
            sb.append(nLayers[i].size());
        }
        sb.append('\n');
        
        sb.append(nLayers[0].getBias());
        for(int i = 1; i < nLayers.length; i++) {
            sb.append(',');
            sb.append(nLayers[i].getBias());
        }
        sb.append("\n");

        return sb.toString();
    }
    
    public abstract void forwardPass(int thread);
    public abstract void calcEvals(Unit next, int thread);

    public abstract ConnectionLayer[] getEntryConnections();
    public abstract NeuronLayer[] getEntryLayers();
    public abstract NeuronLayer[] getExitLayers();
}
