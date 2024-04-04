package neural_plswork.neuron.evaluation.loss;

public enum Loss {
    CUSTOM(0),
    MSE(1),
    BCE(2),
    CE(3),
    INVALID(-1);


    final int id;
    private Loss(int id) {
        this.id = id;
    }

    public Loss getById(int id) {
        switch(id) {
            case 0:
                return CUSTOM;
            case 1:
                return MSE;
            case 2:
                return BCE;
            case 3:
                return CE;
            default:
                return INVALID;
        }
    }

    @Override
    public String toString() {
        return String.valueOf(id);
    }
}
