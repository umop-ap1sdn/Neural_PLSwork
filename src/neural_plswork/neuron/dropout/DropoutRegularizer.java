package neural_plswork.neuron.dropout;

public enum DropoutRegularizer {
    
    CUSTOM(0),
    NONE(1),
    SAMPLE_WISE(2),
    BATCH_WISE(3),
    INVALID(-1);

    final int id;
    private DropoutRegularizer(int id) {
        this.id = id;
    }

    public static DropoutRegularizer getById(int id) {
        switch(id) {
            case 0:
                return CUSTOM;
            case 1:
                return NONE;
            case 2:
                return SAMPLE_WISE;
            case 3:
                return BATCH_WISE;
            default: 
                return INVALID;
        }
    }

    @Override
    public String toString() {
        return String.valueOf(id);
    }
}
