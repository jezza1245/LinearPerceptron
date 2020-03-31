package com.company;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RandomSubset;
import weka.filters.unsupervised.attribute.Remove;

import java.io.EOFException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Random;

public class LinearPerceptronEnsemble implements Classifier, CapabilitiesHandler {

    int ensembleSize;
    LinearPerceptron ensemble[];
    ArrayList<ArrayList<Integer>> attributeIndexes;

    double attributeProportion = 0.5;

    public LinearPerceptronEnsemble(){
        ensembleSize = 50;
        ensemble = new LinearPerceptron[ensembleSize];
        this.initializeEnsemble();
    }

    public void setEnsembleSize(int ensembleSize) { this.ensembleSize = ensembleSize; }

    public void setAttributePortion(double attributePortion){
        this.attributeProportion = attributePortion;
    }

    private void initializeEnsemble(){
        for(int i=0; i<this.ensembleSize; i++){
            ensemble[i] = new LinearPerceptron();
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        instances.setClassIndex(instances.numAttributes()-1);
        attributeIndexes = new ArrayList<>();
        RandomSubset rs = new RandomSubset();
        rs.setNumAttributes(attributeProportion);



        int index = 1;
        for(LinearPerceptron classifier: ensemble){
            Instances newInstances = instances;

            rs.setInputFormat(instances);
            rs.setSeed(index++);


            newInstances = Filter.useFilter(newInstances, rs);

            Enumeration<Attribute> attributeEnumeration = newInstances.enumerateAttributes();
            ArrayList<Integer> relevantAttributesIndexes = new ArrayList<>();
            while(attributeEnumeration.hasMoreElements()){
                relevantAttributesIndexes.add(attributeEnumeration.nextElement().index());
            }
            attributeIndexes.add(relevantAttributesIndexes);

            System.out.println(newInstances);
        }
        System.out.println("done");
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double predictions[] = new double[ensembleSize];

        for(int i=0; i<ensembleSize; i++){

            //Get correct instance by attributes
            Remove remove = new Remove();
            int indices[] = new int[attributeIndexes.get(i).size()];
            for(int j=0; j < indices.length; j++){
                indices[j] = attributeIndexes.get(i).get(j);
            }
            remove.setAttributeIndicesArray(indices);

            Instances instances = new Instances("newInstances",attributes.get(i),0);
            Instance tempInstance = Filter.useFilter(instances,remove).get(0);

            // Classify
            double prediction = ensemble[i].classifyInstance(tempInstance);

            //Add to predictions
            predictions[i] = prediction;
        }

        //Perform majority vote
        int class1 = 0,class2 = 0;
        for(int i=0; i<predictions.length; i++){
            if (predictions[i] == 0.0) {
                class1++;
            } else {
                class2++;
            }
        }

        double finalPrediction = 0.0;
        if(class1 < class2){
            finalPrediction = 1.0;
        }

        return finalPrediction;
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
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        result.setMinimumNumberInstances(0);
        return result;
    }


    public static void main(String[] args) {
        Instances testData = WekaTools.loadClassificationData("resources\\UCIContinuous\\bank\\bank.arff");
        testData.setClassIndex(2);
        LinearPerceptronEnsemble lpe = new LinearPerceptronEnsemble();
        try{
            //lpe.setAttributePortion(0.5);
            lpe.buildClassifier(testData);
        }catch (Exception e){
            e.printStackTrace();
        }
        System.out.println("hello");
    }
}
