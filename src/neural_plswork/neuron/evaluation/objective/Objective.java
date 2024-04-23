package neural_plswork.neuron.evaluation.objective;

public enum Objective {
    CUSTOM(0),
    ROC_AUC(1),
    BINARY_ACCURACY(2),
    ACCURACY(3),
    INVALID(-1);


    final int id;
    private Objective(int id) {
        this.id = id;
    }

    public Objective getById(int id) {
        switch(id) {
            case 0:
                return CUSTOM;
            case 1:
                return ROC_AUC;
            case 2:
                return BINARY_ACCURACY;
            case 3:
                return ACCURACY;
            default:
                return INVALID;
        }
    }

    @Override
    public String toString() {
        return String.valueOf(id);
    }
}
