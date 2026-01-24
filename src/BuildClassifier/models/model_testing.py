
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, auc
from sklearn.metrics import recall_score, precision_score, roc_auc_score, precision_recall_curve, roc_curve
from sklearn.calibration import CalibrationDisplay

from termcolor import colored
import itertools
from loguru import logger

# TODO: bootstrapping -> tranformation is done for every samples -> must be taken from x_test_transform

class ModelTesting:

    def __init__(self, y_test, x_test, pipeline, x_transformed=None):

        self.pipeline = pipeline
        self.y_test = y_test
        self.x_test = x_test
        self.x_transformed = x_transformed

        try:
            self.y_expected_predicted_prob = self._predict_expected_precision()
        except Exception as e:
            logger.warning(f"Model does not have 'predict_expected_precision' method. \n {e}")
            self.y_expected_predicted_prob = None
        
        try:
            self.y_calibrated_prob = self._predict_calibrated_prob()
        except Exception as e:
            logger.warning(f"Model does not have 'predict_calibrated' method. \n {e}")
            self.y_calibrated_prob = None 

        self.y_predicted_prob = pipeline.predict_proba(x_test)[:,1]
        
        # if model is uncertainity aware
        if pipeline[-1].use_beta:
            self.uncert_predicted = self.predict_uncert()

    def _predict_expected_precision(self, x_test=None):
        """
        return expected precision
        """
        if x_test is None:
            x_test = self.x_test
        if self.x_transformed is None:
            self.x_transformed = self.pipeline[:-1].transform(x_test)
        return self.pipeline[-1].predict_expected_precision(self.x_transformed)
    
    def _predict_calibrated_prob(self, x_test=None):
        """
        return calibrated probability
        """
        if x_test is None:
            x_test = self.x_test
        if self.x_transformed is None:
            self.x_transformed = self.pipeline[:-1].transform(self.x_test)
        return self.pipeline[-1].predict_calibrated(self.x_transformed)

    def predict_uncert(self):
        """
        return prediction with uncertainty 
        """
        if self.x_transformed is None:
            self.x_transformed = self.pipeline[:-1].transform(self.x_test)
        return self.pipeline[-1].predict_uncert(self.x_transformed)

    def get_scores(self, y_test=None, cut=0.95, y_predicted_prob=None, prob_type="raw", rounding=4):
        """
        get various classification scores as dictionary

        Args:
            y_test (1D array, optional): test target array. Defaults to None. If not given, take class variable. Need for bootstraping.
            cut (float, optional): threshold cut. Defaults to 0.95.
            y_predicted_prob (2D array, optional): output probability. Defaults to None. If not given, take class variable. Need for bootstraping.
            prob_type (str, optional): type of prob output to use. Must be 'raw', 'precision' or 'calibrated'
            rounding (int, optional): format output. Defaults to 4.
        """

        if y_predicted_prob is None:
            y_predicted_prob = self._return_proba(prob_type)
        if y_test is None:
            y_test = self.y_test

        y_predicted_classes = np.where(y_predicted_prob >= cut, 1, 0)

        # Average Precision Score
        avg_prec_score = round(average_precision_score(y_test, y_predicted_prob), rounding)

        # ROC
        roc_score = round(roc_auc_score(y_test, y_predicted_prob), rounding)

        # Recall Score
        rec_score = round(recall_score(y_test, y_predicted_classes), rounding)

        # precision Score
        prec_score = round(precision_score(y_test, y_predicted_classes), rounding)

       # confusion matrix
        cm = confusion_matrix(y_test, y_predicted_classes)
        
        # False Positives share
        fp_share = round(cm[0][1] / cm.sum(), rounding)

        return {"avg_prec_score":avg_prec_score, "roc_score":roc_score, "rec_score":rec_score, "prec_score":prec_score, "fp_share":fp_share, "cm":cm}
    
    def inspect_proba_distribution(self, prob_type="raw", plot_kwargs={"alpha":0.4, "bins":20, "density":True}, ax=None):
        """
        plot the probability distribution using raw probability output or expected precision output.
        """
        if ax is None:
            ax = plt.gca()

        y_predicted_prob = self._return_proba(prob_type)
        
        # get label
        label = self._get_label(plot_kwargs, prob_type)

        ax.hist(y_predicted_prob, label=label, **plot_kwargs)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Density")
        ax.legend()

        return ax
    
    def inspect_callibration(self):
        """
        plot calibration chart -> maping between probability output and observed frequency in the Test set 
        """

        fig, ax = plt.subplots(figsize=(6, 6))

        CalibrationDisplay.from_predictions(
            self.y_test, self.y_predicted_prob, name="Raw probability output", ax=ax
        )

        CalibrationDisplay.from_predictions(
            self.y_test, self.y_expected_predicted_prob, name="Expected precision", ax=ax
        )

        CalibrationDisplay.from_predictions(
            self.y_test, self.y_calibrated_prob, name="Calibrated probability", ax=ax
        )

        # Optionally customize
        ax.set_title("Calibration Curves")
        plt.legend()
        plt.tight_layout()

        return fig

    def get_score_uncertainity(self, cut=0.95, prob_type="raw", n_samples=50, frac=0.8):
        """
        Performs bootstraping on the Test set and get the scores for each samples.
        Allows to get an estimation on the score uncertainty. 
        """
        
        # recreate df total with Test data
        df_tot = pd.concat([self.x_test, self.y_test], axis=1)
        target_col = self.y_test.name
        
        # array with bootstraped scoring for each of the 5 scores
        boot_metric = np.zeros(shape=(n_samples, 5))

        # loop over bootstraps
        for i in range(n_samples):

            # shuffle with replacement
            boot = df_tot.sample(frac=frac, replace=True)
            boot_x_test, boot_y_test = boot.drop(columns=[target_col], inplace=False), boot[target_col]

            # prediction on sample
            if prob_type == "precision":
                y_pred = self._predict_expected_precision(boot_x_test)
            elif prob_type == "raw":
                y_pred = self.pipeline.predict_proba(boot_x_test)[:,1]
            elif prob_type == "calibrated":
                y_pred = self._predict_calibrated_prob(boot_x_test)
            else:
                err_str = f"Type of probability output not recognized. Got {prob_type}, expected 'raw', 'precision' or 'calibrated'."
                logger.error(err_str)
                raise ValueError(err_str)

            # calculate and store metrics on sample
            res = self.get_scores(cut=cut, y_test=boot_y_test, y_predicted_prob=y_pred)
            boot_metric[i, :] = np.array(list(res.values())[:-1])
        
        return pd.DataFrame(data=boot_metric, columns=["avg_prec_score", "roc_score", "rec_score", "prec_score", "fp_share"])
    
    def get_classification_report(self, cut=0.5, target_names=["non-scam", "scam"]):
        """
        return sklearn's default classification report for a given cutoff
        """
        y_predicted_classes = np.where(self.y_predicted_prob >= cut, 1, 0)
        return classification_report(self.y_test, y_predicted_classes, target_names=target_names)
    
    def inspect_precision_recall_curve(self, prob_type="raw", cutoff_points: list = [.975, .95, .9, .8, .7, .5], plot_kwargs={"color":"blue", "lw":2}, ax=None):
        """
        Plot the precision-recall curve with a no-skill (random) baseline and the cutoff points used for scoring.
        
        Returns:
            matplotlib.axes._subplots.AxesSubplot: the axis object
        """

        y_test = self.y_test
        y_predicted_prob = self._return_proba(prob_type)
        avg_prec = average_precision_score(y_test, y_predicted_prob) *100

        if ax is None:
            ax = plt.gca()

        # get label
        label = self._get_label(plot_kwargs, prob_type)
        
        # add average precision to label
        label= f'{label} (AVG PR = {avg_prec:0.1f}%)'
        
        precision, recall, op_val = precision_recall_curve(y_test, y_predicted_prob)
        df = pd.DataFrame({'precision': precision[:-1], 'recall': recall[:-1], 'op': op_val})
        op_sel = [np.argmin(abs(df.op - op)) for op in cutoff_points]

        ax.plot(df['recall'], df['precision'], label=label, **plot_kwargs)
        ax.plot(df.recall.iloc[op_sel], df.precision.iloc[op_sel], 'k*')

        # no skill baseline
        ax.axhline(y_test.mean(), color='navy', lw=1, linestyle='--', label='no skill baseline')

        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title('Precision-Recall-Curve ')

        # remove duplicate legend
        self._add_legend_wo_duplicate(ax)

        return ax

    def inspect_receiver_operator_curve(self, prob_type="raw", plot_kwargs={"color":"darkorange", "lw":2}, ax=None):
        """
        Plot the Receiver operating characteristic (precision-recall) curve with a no-skill (random) baseline.
        
        Returns:
            matplotlib.axes._subplots.AxesSubplot: the axis object
        """

        y_predicted_prob = self._return_proba(prob_type)
        y_test = self.y_test

        if ax is None:
            ax = plt.gca()

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob)

        fpr, tpr, _ = roc_curve(y_test, y_predicted_prob)
        roc_auc = auc(fpr, tpr)*100

        # get label
        label = self._get_label(plot_kwargs, prob_type)
        
        # add roc auc in label
        label = f'{label} (AUC = {roc_auc:0.1f}%)'

        ax.plot(fpr, tpr, label=label, **plot_kwargs)
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='no skill baseline')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic example')
        self._add_legend_wo_duplicate(ax)

        return ax

    def inspect_confusion_matrix(self, prob_type="raw", cut=0.985, classes_name=["0", "1"],
                                normalize=False,
                                title='Confusion matrix',
                                cmap=plt.cm.Blues,
                                recall_thresholds=(60, 50)):
        
        y_pred_proba = self._return_proba(prob_type)
        y_test = self.y_test

        y_predicted_classes = np.where(y_pred_proba >= cut, 1, 0)
        cm = confusion_matrix(y_test, y_predicted_classes)

        print('Metrics')
        print(colored(f'{classes_name[0]} share (test): {sum(y_test) * 100 / len(y_test):.1f}%', 'grey'))
        print(colored(f'{classes_name[0]}  share (pred): {sum(y_predicted_classes) * 100 / len(y_predicted_classes):.1f}%', 'grey'))

        recall = recall_score(y_test, y_predicted_classes) * 100
        if recall > recall_thresholds[0]:
            recall_color = 'green'
        elif recall > recall_thresholds[1]:
            recall_color = 'yellow'
        else:
            recall_color = 'red'
        print(colored(f'Recall: {recall:.1f}%', recall_color))

        prec = precision_score(y_test, y_predicted_classes) * 100
        if prec > (cut * 100) - 1:
            prec_color = 'green'
        elif prec > (cut * 100) - 3.5:
            prec_color = 'yellow'
        else:
            prec_color = 'red'
        print(colored(f'Precision: {prec:.1f}%', prec_color))

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("\nNormalized confusion matrix")
        else:
            print('\nConfusion matrix, without normalization')

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    
    def _return_proba(self, prob_type="raw"):
        if prob_type == "precision":
            return self.y_expected_predicted_prob
        elif prob_type == "raw":
            return self.y_predicted_prob
        elif prob_type == "calibrated":
            return self.y_calibrated_prob
        else:
            err_str = f"Type of probability output not recognized. Got {prob_type}, expected 'raw', 'precision' or 'calibrated'."
            logger.error(err_str)
            raise ValueError(err_str)
    
    def _get_label(self, plot_kwargs, prob_type):
        if "label" not in plot_kwargs.keys():
            if prob_type == "precision":
                label="expected precision"
            elif prob_type == "raw":
                label="raw output probability"
            elif prob_type == "calibrated":
                label = "calibrated probability"
            else:
                err_str = f"Type of probability output not recognized. Got {prob_type}, expected 'raw', 'precision' or 'calibrated'."
                logger.error(err_str)
                raise ValueError(err_str)
        else:
            label = plot_kwargs["label"]
            del plot_kwargs["label"]
        return label
    
    def _add_legend_wo_duplicate(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

     

