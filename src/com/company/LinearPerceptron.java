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

    private boolean debug = true;
    private DecimalFormat df = new DecimalFormat("#.00");
    private int maxIterations = 20;
    double weights[];

    private void trainPerceptron(Instances instances) throws Exception{

        double n = 1; // n (learning rate) = 1
        double y; // predicted output y
        double t; // actual output

        int numIterations = 0;
        int totalInstances= instances.numInstances();
        int iterationsSinceUpdate = 0;

        boolean revolutionWithoutpdate;
        boolean atIterationLimit;

        do{
            int index = numIterations%totalInstances;
            Instance instance = instances.get(index);

            y = classifyInstance(instance); // Classify the instance
            y = y<0 ? -1 : 1; // Apply Logistic function to map y to -1 (if negative) or 1 (if y >= 0)

            t = instance.classValue(); // Get actual class value

            if(debug) System.out.print(numIterations+"("+index+")    y="+y+" t="+t);

            // If incorrect classification was made
            if(y!=t) {
                if(debug) System.out.print("  Updating weights... "+ df.format(weights[0]) + "," + df.format(weights[1]));

                // Update weights across all attributes
                for (int i = 0; i < instances.numAttributes(); i++) {
                    weights[i] = weights[i] + 0.5 * n * (t - y) * instance.value(i); // Could ignore class value, but has no effect on classification
                }
                iterationsSinceUpdate = 0;

                if(debug) System.out.print("  --> "+ df.format(weights[0]) + "," + df.format(weights[1]));
            }
            else iterationsSinceUpdate++; // no update, increase count of no updates

            numIterations++; //increase count of iterations

            if(debug) System.out.println();

            revolutionWithoutpdate = iterationsSinceUpdate >= (instances.numInstances()-1); // Set flag if a full revolution has been made
            atIterationLimit = numIterations >= maxIterations; // Set flag if at iteration limit

        }while(!revolutionWithoutpdate && !atIterationLimit); //Stopping function

        if(revolutionWithoutpdate) System.out.println("Iterated through all instances without update.");
        if(atIterationLimit) System.out.println("Iteration limit reached");

    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        weights = new double[instances.numAttributes()]; // weights (weights), array of weights for each attribute

        for(int i=0; i< weights.length; i++){
            weights[i] = 1; //(int)(Math.random()*10);
        }

        this.trainPerceptron(instances);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        /*
        The method classifyInstance should applies the model to the new instance then applies
        a sensible decision rule to the resulting linear prediction.
        */
        double prediction_real = 0;
        for(int i = 0; i < weights.length; i++){
            prediction_real = prediction_real + (weights[i] * instance.value(i));
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
            lp.debug = true;
            lp.buildClassifier(testData);
            System.out.println(WekaTools.accuracy(lp, testData));
        }catch (Exception e){
            e.printStackTrace();
        }
    }



}
