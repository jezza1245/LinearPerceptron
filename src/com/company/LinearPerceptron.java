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

    protected boolean debug = true;
    protected DecimalFormat df = new DecimalFormat("#.00");
    protected int maxIterations = 1000000;
    protected double weights[];
    protected double learningRate = 1;


    public void setMaxIterations(int maxIterations) { this.maxIterations = maxIterations; }
    public void setLearningRate(double learningRate) { this.learningRate = learningRate; }

    private void trainPerceptron(Instances instances) throws Exception{

        double y; // predicted output y
        double t; // actual output

        int numIterations = 0;
        int totalInstances= instances.numInstances();
        int iterationsSinceUpdate = 0;

        boolean revolutionWithoutUpdate;
        boolean atIterationLimit;

        do{
            int index = numIterations%totalInstances;
            Instance instance = instances.get(index);

            y = classifyInstance(instance); // Classify the instance
            y = y<0 ? -1 : 1; // Apply Logistic function to map y to -1 (if negative) or 1 (if y >= 0)

            t = instance.classValue(); // Get actual class value
            if(t==0) t=-1; //If class given is 0, set to -1 to work with perceptron logic
            if(debug) System.out.print(numIterations+"("+index+")    y="+y+" t="+t);

            // If incorrect classification was made
            if(y!=t) {

                if(debug) System.out.print("  Updating weights... "+ df.format(weights[0]) + "," + df.format(weights[1]));

                // Update weights across all attributes
                for (int i = 0; i < instances.numAttributes(); i++) {
                    weights[i] = weights[i] + 0.5 * learningRate * (t - y) * instance.value(i); // Could ignore class value, but has no effect on classification
                }
                iterationsSinceUpdate = 0;

                if(debug) System.out.print("  --> "+ df.format(weights[0]) + "," + df.format(weights[1]));
            }
            else iterationsSinceUpdate++; // no update, increase count of no updates

            numIterations++; //increase count of iterations

            if(debug) System.out.println();

            revolutionWithoutUpdate = iterationsSinceUpdate >= (totalInstances-1); // Set flag if a full revolution has been made
            atIterationLimit = numIterations >= maxIterations; // Set flag if at iteration limit

        }while(!revolutionWithoutUpdate && !atIterationLimit); //Stopping function

        if(revolutionWithoutUpdate && debug) System.out.println("Iterated through all instances without update.");
        if(atIterationLimit && debug) System.out.println("Iteration limit reached");

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

        double[] distributions = new double[2];
        double classifiedAs = classifyInstance(instance);
        if(classifiedAs == -1) classifiedAs = 0;
        distributions[(int)classifiedAs] = 1.0;

        return distributions;
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

    private void printClassDistributions(Instances instances) throws Exception {
        for(Instance i: instances){
            System.out.print(i.classValue()+ "   -> ");
            for(double d: distributionForInstance(i)){
                System.out.print(d+",");
            }
            System.out.println();
        }
    }


    public static void main(String[] args) {
//        Instances testData = WekaTools.loadClassificationData("resources\\test_data.arff");
//        testData.setClassIndex(2);
//        try{
//            lp.debug = false;
//
//            lp.buildClassifier(testData);
//            System.out.println(WekaTools.accuracy(lp, testData));
//
//        }catch (Exception e){
//            e.printStackTrace();
//        }

        try{

            Instances blood[] = WekaTools.getDataSetSplit("blood");
            Instances train = blood[0];
            Instances test = blood[1];

            train.setClassIndex(train.numAttributes()-1);
            test.setClassIndex(test.numAttributes()-1);

            LinearPerceptron lp = new LinearPerceptron();
            lp.debug = false;
            lp.buildClassifier(train);

            System.out.println(WekaTools.accuracy(lp,test));

        }catch(Exception e){
            e.printStackTrace();
        }

    }



}
