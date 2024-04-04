package neural_plswork.neuron.activations;

public enum Activation {
    
    CUSTOM(0),
    LINEAR(1),
    RELU(2),
    SIGMOID(3),
    TANH(4),
    SOFTMAX(5),
    INVALID(-1);




    final int id;
    private Activation(int id) {
        this.id = id;
    }

    public static Activation getById(int id) {
        switch(id) {
            case 0:
                return CUSTOM;
            case 1:
                return LINEAR;
            case 2:
                return RELU;
            case 3:
                return SIGMOID;
            case 4:
                return TANH;
            case 5:
                return SOFTMAX;
            default: 
                return INVALID;
        }
    }

    @Override
    public String toString() {
        return String.valueOf(id);
    }
}
