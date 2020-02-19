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

import java.text.DecimalFormat;

public class LinearPerceptron implements Classifier, CapabilitiesHandler {

    boolean debug = true;
    DecimalFormat df = new DecimalFormat("#.00");
    double w[];

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

        w = new double[instances.numAttributes()]; // w (weights), array of weights for each attribute
        for(int i=0; i< w.length; i++){
            w[i] = 1; //(int)(Math.random()*10);
        }
        double n = 1; // n (learning rate) = 1
        double y; // predicted output y
        double t; // actual output

        int max_iterations = 20;
        int num_iterations = 0;

        do{
            for(Instance instance: instances){
                num_iterations++; // Increase iteration count

                y = classifyInstance(instance); // Classify the instance
                y = y<0 ? -1 : 1; // Apply Logistic function to map y to -1 (if negative) or 1 (if y >= 0)

                t = instance.classValue(); // Get actual class value

                if(debug) System.out.print(num_iterations+"("+num_iterations%instances.numInstances()+")    y="+y+" t="+t);
                // If incorrect classification was made
                if(y!=t) {
                    if(debug) System.out.print("  Updating weights... "+ df.format(w[0]) + "," + df.format(w[1]));

                    // Update weights across all attributes
                    for (int i = 0; i < instances.numAttributes(); i++) {
                        w[i] = w[i] + 0.5 * n * (t - y) * instance.value(i); // Could ignore class value, but has no effect on classification
                    }
                    if(debug) System.out.print("  --> "+ df.format(w[0]) + "," + df.format(w[1]));
                }
                if(debug) System.out.println();
            }

        }while(num_iterations < max_iterations); //Stopping function

    }


//    private boolean Stopping(int num_iterations, int max_iterations){
//        return true ? (num_iterations > max_iterations) : false;
//    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        /*
        The method classifyInstance should applies the model to the new instance then applies
        a sensible decision rule to the resulting linear prediction.
        */
        double prediction_real = 0;
        for(int i = 0; i < w.length; i++){
            prediction_real = prediction_real + (w[i] * instance.value(i));
        }
        return prediction_real >= 0 ? 1.0 : -1.0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception { // TODO
        return new double[0];
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = new Capabilities(this);
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        //result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        //result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);
        return result;
    }


    public static void main(String[] args) {
        Instances testData = WekaTools.loadClassificationData("src\\test_data.arff");
        testData.setClassIndex(2);
        LinearPerceptron lp = new LinearPerceptron();
        try{
            lp.buildClassifier(testData);
            //System.out.println(testData);

            //System.out.println(WekaTools.accuracy(lp, testData));
        }catch (Exception e){
            e.printStackTrace();
        }
    }



}
