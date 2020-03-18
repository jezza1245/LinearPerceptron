package com.company;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.FileReader;
import java.util.Random;

public class WekaTools {

    public static Instances loadClassificationData(String filePath){
        Instances data;
        try{
            FileReader reader = new FileReader(filePath);
            data = new Instances(reader);
        }catch(Exception e){
            System.out.println("Exception caught: "+e);
            data = null;
        }
        return data;
    }

    public static double crossValError(Classifier classifier, Instances instances) throws Exception{

        Evaluation crossValidate = new Evaluation(instances);
        int numFolds = instances.numInstances();
        crossValidate.crossValidateModel(classifier, instances, numFolds, new java.util.Random(1));

        return crossValidate.errorRate();
    }

    public static Instances standardize(Instances instances) throws Exception {
        Standardize standardize = new Standardize();
        standardize.setInputFormat(instances);

        //Standardise to zero mean and unit standard deviation
        return Filter.useFilter(instances, standardize);
    }

    public static Instances[] splitData(Instances all, double proportion){

        Instances split[] = new Instances[2];
        int totalInstances = all.numInstances();

        int splitAt = (int) (proportion*totalInstances);

        all.randomize(new Random(42));
        split[0] = new Instances(all, 0, splitAt);
        split[1] = new Instances(all, splitAt, totalInstances-splitAt);

        return split;
    }

    public static double accuracy(Classifier classifier, Instances test) {
        int totalInstances = test.numInstances();
        int numCorrect = 0;

        Attribute classAttribute = test.classAttribute();
        for (Instance instance : test) {
            try {
                double predicted = classifier.classifyInstance(instance);
                if(predicted==-1) predicted = 0;
                double actual = instance.value(classAttribute);
                if(actual==-1) actual = 0;

                if (predicted == actual) {
                    numCorrect++;
                }
            } catch (Exception e) {
                System.out.println("Error classifying instance");

            }
        }

        double acc = numCorrect/(double)totalInstances;

        return acc;
    }

    public static void main(String[] args) {

    }


}
