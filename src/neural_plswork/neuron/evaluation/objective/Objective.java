package neural_plswork.neuron.evaluation.objective;

public enum Objective {
    CUSTOM(0),
    AUC(1),
    ACCURACY(2),
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
                return AUC;
            case 2:
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
