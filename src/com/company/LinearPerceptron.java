package com.company;
import weka.classifiers.Classifier;

/*
A standard on-line Linear Perceptron classifier implemening the Weka Classifier
interface (or implemening AbstractClassifier).
Works when all attributes are continuous
If this is not the case it will throw an exception.
 */

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;

public class LinearPerceptron implements Classifier, CapabilitiesHandler {
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        /*
        The method buildClassifier creates the linear model by iterating over the training data and applying the on-line
        rule as seen below...

            initialise w to random values
            initialise learning rate η
                do
                    for i=1 to n
                    yi = ψ(w, xi)
                    for j=1 to m
                        ∆wj ← 0.5η(ti − yi)xij
                        wj ← wj + ∆wj
                while (Stopping(t, y) == false)
            return w

        ... cycle through training patterns, if a pattern is currently misclassified
        add/subtract the input vector to the weights (This shifts the discriminant to make it more likely to classify
        that pattern correctly next time).

        It allows for the possible inclusion of a constant (bias) term and include a stopping condition, such as the number of
        iterations to perform.
        */

    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        /*
        The method classifyInstance should applies the model to the new instance then applies
        a sensible decision rule to the resulting linear prediction.
        */
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        //result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);

        //result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);
        return result;
    }






}
