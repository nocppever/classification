import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import truncnorm
import warnings
warnings.filterwarnings('ignore')
import json
import os

class WettabilityDataGenerator:
    def __init__(self):
        """Initialize the data generator with material properties and treatments"""
        self.materials = {
            'Glass': {
                'surface_energy_range': (70, 80),  # mN/m
                'thermal_conductivity_range': (0.8, 1.2),  # W/(m·K)
                'youngs_modulus_range': (65, 75),  # GPa
                'density_range': (2.4, 2.8),  # g/cm³
                'base_contact_angle_range': (20, 40)  # degrees
            },
            'PTFE': {
                'surface_energy_range': (18, 22),
                'thermal_conductivity_range': (0.23, 0.27),
                'youngs_modulus_range': (0.4, 0.6),
                'density_range': (2.1, 2.2),
                'base_contact_angle_range': (108, 122)
            },
            'Stainless_Steel': {
                'surface_energy_range': (40, 45),
                'thermal_conductivity_range': (14, 16),
                'youngs_modulus_range': (190, 210),
                'density_range': (7.7, 8.0),
                'base_contact_angle_range': (70, 85)
            },
            'PMMA': {
                'surface_energy_range': (35, 40),
                'thermal_conductivity_range': (0.17, 0.22),
                'youngs_modulus_range': (2.4, 3.1),
                'density_range': (1.17, 1.20),
                'base_contact_angle_range': (67, 80)
            },
            'Silicon_Wafer': {
                'surface_energy_range': (45, 55),
                'thermal_conductivity_range': (130, 150),
                'youngs_modulus_range': (130, 170),
                'density_range': (2.32, 2.33),
                'base_contact_angle_range': (35, 45)
            }
        }

        self.surface_treatments = {
            'Untreated': {
                'angle_modifier': (-5, 10),  
                'roughness_range': (0.1, 0.5)  
            },
            'Plasma': {
                'angle_modifier': (-45, -15),  
                'roughness_range': (0.05, 0.4)
            },
            'Silane': {
                'angle_modifier': (25, 50),  
                'roughness_range': (0.2, 0.7)
            },
            'Fluorination': {
                'angle_modifier': (55, 85),  
                'roughness_range': (0.3, 1.0)
            },
            'UV_Ozone': {
                'angle_modifier': (-35, -10),  
                'roughness_range': (0.1, 0.5)
            }
        }

        self.environmental_conditions = {
            'temperature_range': (18, 27),  
            'humidity_range': (35, 65),     
            'pressure_range': (990, 1020)   
        }

    def generate_data(self, n_samples=1000):
        """Generate synthetic wettability data with realistic noise and variations"""
        data = []
        
        for _ in range(n_samples):
            ambient_temp = np.random.uniform(*self.environmental_conditions['temperature_range'])
            ambient_humidity = np.random.uniform(*self.environmental_conditions['humidity_range'])
            ambient_pressure = np.random.uniform(*self.environmental_conditions['pressure_range'])
            
            day_variation = np.random.normal(0, 1)
            
            material = np.random.choice(list(self.materials.keys()))
            treatment = np.random.choice(list(self.surface_treatments.keys()))
            
            material_props = self.materials[material]
            treatment_props = self.surface_treatments[treatment]
            
            measurement_noise = np.random.normal(0, 0.05)  
            surface_variation = np.random.normal(0, 0.1)   
            
            sample = {
                'material': material,
                'surface_treatment': treatment,
                'surface_energy': np.random.uniform(*material_props['surface_energy_range']) * (1 + measurement_noise),
                'thermal_conductivity': np.random.uniform(*material_props['thermal_conductivity_range']) * (1 + measurement_noise),
                'youngs_modulus': np.random.uniform(*material_props['youngs_modulus_range']) * (1 + measurement_noise),
                'density': np.random.uniform(*material_props['density_range']) * (1 + measurement_noise),
                'roughness_ra': np.random.uniform(*treatment_props['roughness_range']) * (1 + surface_variation)
            }
            
            base_porosity = 0.1 if material in ['Glass', 'Silicon_Wafer'] else 0.2
            treatment_factor = 1.5 if treatment in ['Plasma', 'UV_Ozone'] else 1.0
            humidity_effect = (ambient_humidity - 50) / 100  
            sample['porosity'] = min(0.6, max(0.01, base_porosity * treatment_factor * 
                                            (1 + humidity_effect + measurement_noise)))
            
            base_angle = np.random.uniform(*material_props['base_contact_angle_range'])
            angle_modifier = np.random.uniform(*treatment_props['angle_modifier'])
            
            temp_effect = (ambient_temp - 22) * 0.5  
            humidity_effect = (ambient_humidity - 50) * 0.3  
            pressure_effect = (ambient_pressure - 1013) * 0.1  
            
            contact_angle = (base_angle + 
                           angle_modifier + 
                           temp_effect + 
                           humidity_effect + 
                           pressure_effect + 
                           np.random.normal(0, 5) +  
                           day_variation)  
            
            surface_energy_factor = 1 + (sample['surface_energy'] - 50) / 200
            contact_angle *= (surface_energy_factor + measurement_noise)
            
            contact_angle = max(0, min(180, contact_angle))
            sample['contact_angle'] = contact_angle
            
            sample.update({
                'temperature': ambient_temp,
                'humidity': ambient_humidity,
                'pressure': ambient_pressure
            })
            
            if contact_angle < 10:
                prob_super = 0.9
                prob_hydro = 0.1
                prob_dict = {
                    'Superhydrophilic': prob_super,
                    'Hydrophilic': prob_hydro
                }
            elif contact_angle < 90:

                if contact_angle < 15:
                    prob_super = 0.3
                    prob_hydro = 0.7
                    prob_dict = {
                        'Superhydrophilic': prob_super,
                        'Hydrophilic': prob_hydro
                    }

                elif contact_angle > 85:
                    prob_hydro = 0.7
                    prob_phobic = 0.3
                    prob_dict = {
                        'Hydrophilic': prob_hydro,
                        'Hydrophobic': prob_phobic
                    }
                else:
                    prob_dict = {'Hydrophilic': 1.0}
            elif contact_angle < 150:

                if contact_angle > 145:
                    prob_phobic = 0.7
                    prob_super = 0.3
                    prob_dict = {
                        'Hydrophobic': prob_phobic,
                        'Superhydrophobic': prob_super
                    }
                else:
                    prob_dict = {'Hydrophobic': 1.0}
            else:
                prob_dict = {'Superhydrophobic': 1.0}
            
            sample['wettability_class'] = np.random.choice(
                list(prob_dict.keys()),
                p=list(prob_dict.values())
            )
            
            data.append(sample)
        
        return pd.DataFrame(data)

class MultiModelWettabilityClassifier:
    def __init__(self):
        self.numerical_features = [
            'contact_angle', 'surface_energy', 'roughness_ra', 
            'porosity', 'thermal_conductivity', 'youngs_modulus', 
            'density', 'temperature', 'humidity', 'pressure'
        ]
        self.categorical_features = ['material', 'surface_treatment']
        self.target = 'wettability_class'
        
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [150, 200, 250],
                    'max_depth': [8, 10, 12],
                    'min_samples_split': [2, 3],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt'],  
                    'class_weight': ['balanced', None]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    random_state=42,
                    eval_metric='mlogloss',
                    enable_categorical=True
                ),
                'params': {
                    'n_estimators': [150, 200],
                    'max_depth': [3, 4],
                    'learning_rate': [0.01, 0.05],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9],
                    'gamma': [0, 0.1],
                    'min_child_weight': [1, 3]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [150, 200],
                    'max_depth': [3, 4],
                    'learning_rate': [0.01, 0.05],
                    'subsample': [0.8, 0.9],
                    'max_features': ['sqrt'],
                    'min_samples_split': [2, 3]
                }
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(
                    random_state=42,
                    verbose=-1,
                    min_split_gain=0.01
                ),
                'params': {
                    'n_estimators': [150, 200],
                    'max_depth': [3, 4],
                    'learning_rate': [0.01, 0.05],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8],
                    'min_child_weight': [3, 5],
                    'num_leaves': [15, 31]
                }
            },
            'neural_network': {
                'model': MLPClassifier(
                    random_state=42,
                    max_iter=1000,
                    early_stopping=True,  
                    validation_fraction=0.1
                ),
                'params': {
                    'hidden_layer_sizes': [(100, 50), (150, 75)],
                    'activation': ['tanh'],  
                    'alpha': [0.001, 0.0005],
                    'learning_rate_init': [0.001],
                    'batch_size': [32, 64]
                }
            },
            'svm': {
                'model': SVC(
                    random_state=42,
                    probability=True,
                    cache_size=1000  
                ),
                'params': {
                    'C': [5.0, 10.0],
                    'kernel': ['rbf'],  
                    'gamma': ['scale'],
                    'class_weight': [None]  
                }
            }
        }
        
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Validate that all required libraries are available"""
        missing_deps = []
        try:
            import xgboost
        except ImportError:
            missing_deps.append("xgboost")
        try:
            import lightgbm
        except ImportError:
            missing_deps.append("lightgbm")
            
        if missing_deps:
            print(f"Warning: The following dependencies are missing and their models will be skipped: {', '.join(missing_deps)}")
            for dep in missing_deps:
                if dep.lower() in self.model_configs:
                    del self.model_configs[dep.lower()]
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data with proper encoding and scaling"""
        data = df.copy()
        
        expected_columns = self.numerical_features + self.categorical_features
        if not all(col in data.columns for col in expected_columns):
            raise ValueError(f"Missing columns in input data. Expected: {expected_columns}")
        
        if data[self.numerical_features].isnull().any().any():
            print("Warning: Numerical features contain missing values. Filling with median.")
            data[self.numerical_features] = data[self.numerical_features].fillna(
                data[self.numerical_features].median()
            )
        
        if data[self.categorical_features].isnull().any().any():
            print("Warning: Categorical features contain missing values. Filling with mode.")
            data[self.categorical_features] = data[self.categorical_features].fillna(
                data[self.categorical_features].mode().iloc[0]
            )
        
        for feature in self.categorical_features:
            if is_training:
                self.label_encoders[feature] = LabelEncoder()
                data[feature] = self.label_encoders[feature].fit_transform(data[feature])
            else:

                unique_values = set(data[feature].unique())
                known_values = set(self.label_encoders[feature].classes_)
                unknown_values = unique_values - known_values
                if unknown_values:
                    print(f"Warning: Unknown categories in {feature}: {unknown_values}")
                    mode_category = self.label_encoders[feature].transform([self.label_encoders[feature].classes_[0]])[0]
                    data.loc[data[feature].isin(unknown_values), feature] = mode_category
                data[feature] = self.label_encoders[feature].transform(data[feature])
        
        if is_training:
            data[self.numerical_features] = self.scaler.fit_transform(data[self.numerical_features])
        else:
            data[self.numerical_features] = self.scaler.transform(data[self.numerical_features])
        
        X = data[self.numerical_features + self.categorical_features]
        
        if self.target in data.columns:
            if is_training:
                y = self.target_encoder.fit_transform(data[self.target])
            else:

                unique_classes = set(data[self.target].unique())
                known_classes = set(self.target_encoder.classes_)
                unknown_classes = unique_classes - known_classes
                if unknown_classes:
                    raise ValueError(f"Unknown classes encountered: {unknown_classes}")
                y = self.target_encoder.transform(data[self.target])
            return X, y
        return X
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train and evaluate all models with cross-validation monitoring"""
        results = []
        failed_models = []
        
        for model_name, config in self.model_configs.items():
            print(f"\nTraining {model_name}...")
            try:

                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5,
                    scoring=['f1_weighted', 'accuracy'],
                    refit='f1_weighted',
                    n_jobs=-1,
                    verbose=1,
                    error_score='raise',
                    return_train_score=True  
                )
                
                grid_search.fit(X_train, y_train)
                
                self.models[model_name] = grid_search.best_estimator_
                
                val_pred = grid_search.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_pred)
                val_f1 = f1_score(y_val, val_pred, average='weighted')
                
                train_pred = grid_search.predict(X_train)
                train_accuracy = accuracy_score(y_train, train_pred)
                train_f1 = f1_score(y_train, train_pred, average='weighted')
                
                results.append({
                    'model': model_name,
                    'best_params': grid_search.best_params_,
                    'cv_score': grid_search.best_score_,
                    'val_accuracy': val_accuracy,
                    'val_f1': val_f1,
                    'train_accuracy': train_accuracy,
                    'train_f1': train_f1,
                    'overfitting_gap': train_f1 - val_f1
                })
                
                print(f"{model_name} Results:")
                print(f"Validation - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
                print(f"Training - Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
                print(f"Overfitting Gap: {train_f1 - val_f1:.4f}")
            
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                failed_models.append(model_name)
                continue
        
        if failed_models:
            print(f"\nWarning: The following models failed to train: {', '.join(failed_models)}")
        
        results_df = pd.DataFrame(results)
        self.best_model_name = results_df.loc[results_df['val_f1'].idxmax(), 'model']
        self.best_model = self.models[self.best_model_name]
        
        return results_df
    

    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train and evaluate all models"""
        results = []
        failed_models = []
        
        for model_name, config in self.model_configs.items():
            print(f"\nTraining {model_name}...")
            try:
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=1,
                    error_score='raise'
                )
                
                grid_search.fit(X_train, y_train)
                self.models[model_name] = grid_search.best_estimator_
                
                val_pred = grid_search.predict(X_val)
                val_accuracy = accuracy_score(y_val, val_pred)
                val_f1 = f1_score(y_val, val_pred, average='weighted')
                
                results.append({
                    'model': model_name,
                    'best_params': grid_search.best_params_,
                    'cv_score': grid_search.best_score_,
                    'val_accuracy': val_accuracy,
                    'val_f1': val_f1
                })
                
                print(f"{model_name} - Validation Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
            
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                failed_models.append(model_name)
                continue
        
        if failed_models:
            print(f"\nWarning: The following models failed to train: {', '.join(failed_models)}")
        
        if not results:
            raise ValueError("No models were successfully trained!")
        
        results_df = pd.DataFrame(results)
        self.best_model_name = results_df.loc[results_df['val_f1'].idxmax(), 'model']
        self.best_model = self.models[self.best_model_name]
        
        return results_df
    
    def predict(self, X):
        """Make predictions and return original class labels"""
        numeric_predictions = self.best_model.predict(X)
        return self.target_encoder.inverse_transform(numeric_predictions)
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models on test set"""
        evaluation_results = []
        
        for model_name, model in self.models.items():
            predictions = model.predict(X_test)
            
            evaluation_results.append({
                'model': model_name,
                'accuracy': accuracy_score(y_test, predictions),
                'f1': f1_score(y_test, predictions, average='weighted'),
                'classification_report': classification_report(
                    self.target_encoder.inverse_transform(y_test),
                    self.target_encoder.inverse_transform(predictions)
                )
            })
        
        return pd.DataFrame(evaluation_results)
    
    def create_evaluation_visualizations(self, X_test, y_test):
        """Create and save detailed evaluation visualizations"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        n_models = len(self.models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
        if n_models > 1:
            axes = axes.ravel()
        else:
            axes = [axes]
        
        class_names = self.target_encoder.classes_
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            predictions = model.predict(X_test)
            cm = confusion_matrix(y_test, predictions)
            
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx],
                       xticklabels=class_names, yticklabels=class_names)
            axes[idx].set_title(f'{model_name} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        for idx in range(len(self.models), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(f'confusion_matrices_{timestamp}.png')
        plt.close()
        
        self._create_feature_importance_plot(timestamp)
        
        self._create_roc_curves_plot(X_test, y_test, timestamp)
        
        self._create_prediction_distribution_plot(X_test, y_test, timestamp)
    
    def _create_feature_importance_plot(self, timestamp):
        """Create feature importance comparison plot"""
        plt.figure(figsize=(12, 6))
        feature_names = self.numerical_features + self.categorical_features
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
        
        if importance_dict:
            importance_df = pd.DataFrame(importance_dict, index=feature_names)
            importance_df.plot(kind='bar')
            plt.title('Feature Importance Comparison')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'feature_importance_{timestamp}.png')
            plt.close()
    
    def _create_roc_curves_plot(self, X_test, y_test, timestamp):
        """Create ROC curves for all models"""
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        plt.figure(figsize=(10, 8))
        
        n_classes = len(self.target_encoder.classes_)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                try:

                    y_score = model.predict_proba(X_test)
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    

                    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    
                    plt.plot(fpr["micro"], tpr["micro"],
                            label=f'{model_name} (area = {roc_auc["micro"]:0.2f})')
                except Exception as e:
                    print(f"Could not create ROC curve for {model_name}: {str(e)}")
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'roc_curves_{timestamp}.png')
        plt.close()
    
    def _create_prediction_distribution_plot(self, X_test, y_test, timestamp):
        """Create prediction distribution plot"""
        plt.figure(figsize=(12, 6))
        
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = self.target_encoder.inverse_transform(model.predict(X_test))
        
        true_labels = self.target_encoder.inverse_transform(y_test)
        
        pred_df = pd.DataFrame(predictions)
        pred_df['True'] = true_labels
        
        pred_df.apply(pd.value_counts).plot(kind='bar')
        plt.title('Prediction Distribution Across Models')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'prediction_distribution_{timestamp}.png')
        plt.close()
    
    def save_models(self, base_filename='wettability_classifier'):
        """Save all trained models and preprocessing components"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_data = {
            'models': self.models,
            'best_model_name': self.best_model_name,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'scaler': self.scaler,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }
        
        filename = f'{base_filename}_{timestamp}.joblib'
        joblib.dump(model_data, filename)
        print(f"Models saved to {filename}")
    
    def load_models(self, filename):
        """Load trained models and preprocessing components"""
        model_data = joblib.load(filename)
        self.models = model_data['models']
        self.best_model_name = model_data['best_model_name']
        self.best_model = self.models[self.best_model_name]
        self.label_encoders = model_data['label_encoders']
        self.target_encoder = model_data['target_encoder']
        self.scaler = model_data['scaler']
        self.numerical_features = model_data['numerical_features']
        self.categorical_features = model_data['categorical_features']
        print("Models loaded successfully")

def create_dataset(n_samples=14000, save_to_csv=False):
    """Create a dataset with the data generator"""
    generator = WettabilityDataGenerator()
    df = generator.generate_data(n_samples)
    
    if save_to_csv:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'synthetic_wettability_data_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    
    return df

def train_and_compare(n_samples=14000):
    """Main function to train and compare all models"""
    print("Generating dataset...")
    df = create_dataset(n_samples=n_samples)
    
    print("\nInitializing multi-model classifier...")
    classifier = MultiModelWettabilityClassifier()
    
    print("\nSplitting dataset...")
    train_val, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    
    print("Preprocessing data...")
    X_train, y_train = classifier.preprocess_data(train, is_training=True)
    X_val, y_val = classifier.preprocess_data(val, is_training=False)
    X_test, y_test = classifier.preprocess_data(test, is_training=False)
    
    print("\nTraining all models...")
    training_results = classifier.train_all_models(X_train, y_train, X_val, y_val)
    
    print("\nPerforming final evaluation...")
    test_results = classifier.evaluate_all_models(X_test, y_test)
    
    print("\nCreating evaluation visualizations...")
    classifier.create_evaluation_visualizations(X_test, y_test)
    
    # Save the models
    classifier.save_models()
    
    return classifier, training_results, test_results


class WettabilityEnsembleAnalyzer:
    def __init__(self, classifier):
        """Initialize with trained classifier instance"""
        self.classifier = classifier
        self.ensemble = None
        self.feature_importances = None
        
    def create_voting_ensemble(self, X_train, y_train):
        """Create a voting ensemble from the top performing models"""
        from sklearn.ensemble import VotingClassifier
        
        top_models = ['random_forest', 'gradient_boosting', 'lightgbm']
        estimators = []
        
        for name in top_models:
            if name in self.classifier.models:
                estimators.append((name, self.classifier.models[name]))
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  
            n_jobs=-1
        )
        
        self.ensemble.fit(X_train, y_train)
        return self.ensemble
    
    def analyze_feature_importance(self):
        """Analyze feature importance across different models"""
        feature_names = (self.classifier.numerical_features + 
                        self.classifier.categorical_features)
        importance_dict = {}
        
        for model_name, model in self.classifier.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
        
        self.feature_importances = pd.DataFrame(importance_dict, index=feature_names)
        
        self.feature_importances['mean_importance'] = self.feature_importances.mean(axis=1)
        self.feature_importances = self.feature_importances.sort_values('mean_importance', ascending=False)
        
        return self.feature_importances
    
    def plot_feature_importance(self):
        """Create detailed feature importance visualizations"""
        if self.feature_importances is None:
            self.analyze_feature_importance()
        
        plt.figure(figsize=(15, 8))
        
        top_features = self.feature_importances.head(10)
        
        sns.heatmap(top_features.drop('mean_importance', axis=1).T,
                   annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Top 10 Feature Importance Across Models')
        plt.xlabel('Features')
        plt.ylabel('Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'feature_importance_heatmap_{timestamp}.png')
        plt.close()
    
    def analyze_predictions(self, X_test, y_test):
        """Analyze predictions from all models with proper type conversion"""
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.classifier.models.items():
            predictions[model_name] = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                probabilities[model_name] = model.predict_proba(X_test)
        
        if self.ensemble is not None:
            predictions['ensemble'] = self.ensemble.predict(X_test)
            probabilities['ensemble'] = self.ensemble.predict_proba(X_test)
        
        report = {
            'model_agreement': {
                k: float(v) for k, v in self._analyze_model_agreement(predictions).items()
            },
            'misclassification_analysis': self._analyze_misclassifications(predictions, y_test),
            'confidence_analysis': self._analyze_prediction_confidence(probabilities, y_test)
        }
        
        return report
    
    def _analyze_model_agreement(self, predictions):
        """Analyze agreement between models"""
        n_samples = len(next(iter(predictions.values())))
        agreement_counts = []
        
        for i in range(n_samples):
            pred_set = set(pred[i] for pred in predictions.values())
            agreement_counts.append(len(pred_set))
        
        return {
            'full_agreement': sum(count == 1 for count in agreement_counts) / n_samples,
            'partial_agreement': sum(count == 2 for count in agreement_counts) / n_samples,
            'disagreement': sum(count > 2 for count in agreement_counts) / n_samples
        }
    
    def _analyze_misclassifications(self, predictions, y_true):
        """Analyze cases where models make mistakes with proper type conversion"""
        misclassification_analysis = {}
        
        for model_name, preds in predictions.items():
            mistakes = y_true != preds
            misclassified_indices = np.where(mistakes)[0]
            
            misclassification_analysis[model_name] = {
                'total_mistakes': int(sum(mistakes)),
                'mistake_rate': float(sum(mistakes) / len(y_true)),
                'misclassified_classes': {
                    str(k): int(v) 
                    for k, v in pd.Series(y_true[mistakes]).value_counts().to_dict().items()
                }
            }
        
        return misclassification_analysis
    
    def _analyze_prediction_confidence(self, probabilities, y_true):
        """Analyze prediction confidence with proper type conversion"""
        confidence_analysis = {}
        
        for model_name, probs in probabilities.items():
            pred_confidence = np.max(probs, axis=1)
            predictions = np.argmax(probs, axis=1)
            correct = predictions == y_true
            
            confidence_analysis[model_name] = {
                'mean_confidence_correct': float(pred_confidence[correct].mean()),
                'mean_confidence_incorrect': float(pred_confidence[~correct].mean()),
                'high_confidence_mistakes': int(sum((pred_confidence > 0.9) & ~correct))
            }
        
        return confidence_analysis

def analyze_wettability_classification(classifier, X_train, y_train, X_test, y_test):
    """Run complete analysis of the classification results"""
    analyzer = WettabilityEnsembleAnalyzer(classifier)
    
    print("\nCreating ensemble classifier...")
    ensemble = analyzer.create_voting_ensemble(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
    
    print(f"Ensemble Performance - Accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}")
    
    print("\nAnalyzing feature importance...")
    feature_importance = analyzer.analyze_feature_importance()
    analyzer.plot_feature_importance()
    
    print("\nAnalyzing predictions...")
    prediction_analysis = analyzer.analyze_predictions(X_test, y_test)
    
    return {
        'ensemble_performance': {
            'accuracy': ensemble_accuracy,
            'f1_score': ensemble_f1
        },
        'feature_importance': feature_importance,
        'prediction_analysis': prediction_analysis
    }

def train_compare_and_analyze():
    """Train models and perform detailed analysis"""
    print("Generating dataset...")
    df = create_dataset(n_samples=14000)
    
    print("\nInitializing multi-model classifier...")
    classifier = MultiModelWettabilityClassifier()
    
    print("\nSplitting dataset...")
    train_val, test = train_test_split(df, test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    
    print("Preprocessing data...")
    X_train, y_train = classifier.preprocess_data(train, is_training=True)
    X_val, y_val = classifier.preprocess_data(val, is_training=False)
    X_test, y_test = classifier.preprocess_data(test, is_training=False)
    
    print("\nTraining all models...")
    training_results = classifier.train_all_models(X_train, y_train, X_val, y_val)
    test_results = classifier.evaluate_all_models(X_test, y_test)
    
    analysis_results = analyze_wettability_classification(
        classifier, X_train, y_train, X_test, y_test
    )
    
    return classifier, training_results, test_results, analysis_results, X_test, y_test

def print_analysis_results(analysis):
    """Print analysis results in a formatted way"""
    print("\nDetailed Analysis Results:")
    
    print("\nEnsemble Performance:")
    print(f"Accuracy: {analysis['ensemble_performance']['accuracy']:.4f}")
    print(f"F1 Score: {analysis['ensemble_performance']['f1_score']:.4f}")
    
    print("\nTop 5 Most Important Features:")
    print(analysis['feature_importance'].head().to_string())
    
    print("\nModel Agreement Analysis:")
    for key, value in analysis['prediction_analysis']['model_agreement'].items():
        print(f"{key}: {value:.4f}")
    
    print("\nMisclassification Analysis:")
    for model_name, results in analysis['prediction_analysis']['misclassification_analysis'].items():
        print(f"\n{model_name}:")
        print(f"  Total mistakes: {results['total_mistakes']}")
        print(f"  Mistake rate: {results['mistake_rate']:.4f}")
        print("  Misclassified classes:")
        for class_name, count in results['misclassified_classes'].items():
            print(f"    {class_name}: {count}")
    
    print("\nConfidence Analysis:")
    for model_name, results in analysis['prediction_analysis']['confidence_analysis'].items():
        print(f"\n{model_name}:")
        print(f"  Mean confidence (correct): {results['mean_confidence_correct']:.4f}")
        print(f"  Mean confidence (incorrect): {results['mean_confidence_incorrect']:.4f}")
        print(f"  High confidence mistakes: {results['high_confidence_mistakes']}")


class WettabilityAdvancedAnalyzer(WettabilityEnsembleAnalyzer):
    def add_advanced_analysis(self, X_test, y_test):
        """Add advanced analysis including calibration and confusion patterns"""
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_dir = f'analysis_viz_{timestamp}'
        os.makedirs(viz_dir, exist_ok=True)
        
        classes = self.classifier.target_encoder.classes_
        n_classes = len(classes)
        
        plt.figure(figsize=(12, 8))
        for model_name, model in self.classifier.models.items():
            if hasattr(model, 'predict_proba'):
                # Compute ROC curve and ROC area for each class
                y_test_bin = label_binarize(y_test, classes=range(n_classes))
                y_score = model.predict_proba(X_test)
                
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                mean_tpr = np.mean([np.interp(all_fpr, fpr[i], tpr[i]) for i in range(n_classes)], axis=0)
                mean_auc = auc(all_fpr, mean_tpr)
                
                plt.plot(all_fpr, mean_tpr, label=f'{model_name} (AUC = {mean_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend(loc="lower right")
        plt.savefig(f'{viz_dir}/roc_curves.png')
        plt.close()

        for model_name, model in self.classifier.models.items():
            plt.figure(figsize=(10, 8))
            predictions = model.predict(X_test)
            cm = confusion_matrix(y_test, predictions, normalize='true')
            
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
            plt.title(f'{model_name} Normalized Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/{model_name}_confusion_matrix.png')
            plt.close()

        if self.feature_importances is not None:
            plt.figure(figsize=(12, 8))
            importance_corr = self.feature_importances.drop('mean_importance', axis=1).corr()
            sns.heatmap(importance_corr, annot=True, fmt='.2f', cmap='RdBu')
            plt.title('Feature Importance Correlation Between Models')
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/feature_importance_correlation.png')
            plt.close()

        misclassified_samples = {}
        all_mistakes = set()
        
        for model_name, model in self.classifier.models.items():
            predictions = model.predict(X_test)
            mistakes = np.where(predictions != y_test)[0]
            misclassified_samples[model_name] = set(mistakes)
            all_mistakes.update(mistakes)
        
        common_mistakes = set.intersection(*misclassified_samples.values())
        unique_mistakes = {
            model_name: mistakes - set.union(*(v for k, v in misclassified_samples.items() if k != model_name))
            for model_name, mistakes in misclassified_samples.items()
        }

        feature_analysis = {}
        for i, feature in enumerate(self.classifier.numerical_features):
            misclassified_values = X_test_array[list(all_mistakes), i]
            correct_values = X_test_array[list(set(range(len(y_test))) - all_mistakes), i]
            
            feature_analysis[feature] = {
                'misclassified': {
                    'mean': float(np.mean(misclassified_values)),
                    'std': float(np.std(misclassified_values)),
                    'min': float(np.min(misclassified_values)),
                    'max': float(np.max(misclassified_values))
                },
                'correct': {
                    'mean': float(np.mean(correct_values)),
                    'std': float(np.std(correct_values)),
                    'min': float(np.min(correct_values)),
                    'max': float(np.max(correct_values))
                }
            }

            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=misclassified_values, label='Misclassified', color='red')
            sns.kdeplot(data=correct_values, label='Correct', color='blue')
            plt.title(f'Distribution of {feature} for Correct vs Misclassified Samples')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(f'{viz_dir}/{feature}_distribution.png')
            plt.close()

        class_performance = {}
        for model_name, model in self.classifier.models.items():
            predictions = model.predict(X_test)
            class_performance[model_name] = {}
            
            for i, class_name in enumerate(classes):
                mask = y_test == i
                class_pred = predictions[mask]
                class_performance[model_name][class_name] = {
                    'accuracy': float(np.mean(class_pred == y_test[mask])),
                    'samples': int(np.sum(mask)),
                    'mistakes': int(np.sum(class_pred != y_test[mask]))
                }

        return {
            'class_performance': class_performance,
            'misclassification_analysis': {
                'total_samples': len(y_test),
                'total_mistakes': len(all_mistakes),
                'common_mistakes': len(common_mistakes),
                'unique_mistakes': {k: len(v) for k, v in unique_mistakes.items()},
                'mistake_overlap': float(len(all_mistakes) / len(y_test))
            },
            'feature_analysis': feature_analysis,
            'visualization_directory': viz_dir
        }
        
    def print_advanced_analysis(advanced_results):
        """Print comprehensive analysis results"""
        print("\nDETAILED CLASSIFICATION ANALYSIS")
        print("=" * 50)
        
        print("\n1. OVERALL MISCLASSIFICATION ANALYSIS")
        print("-" * 40)
        ma = advanced_results['misclassification_analysis']
        print(f"Total samples: {ma['total_samples']}")
        print(f"Total misclassified: {ma['total_mistakes']} ({ma['mistake_overlap']*100:.2f}%)")
        print(f"Common mistakes across all models: {ma['common_mistakes']}")
        print("\nUnique mistakes per model:")
        for model, count in ma['unique_mistakes'].items():
            print(f"  {model}: {count}")
        
        print("\n2. CLASS-WISE PERFORMANCE")
        print("-" * 40)
        for model_name, class_perf in advanced_results['class_performance'].items():
            print(f"\n{model_name}:")
            for class_name, metrics in class_perf.items():
                print(f"  {class_name}:")
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                print(f"    Mistakes: {metrics['mistakes']}/{metrics['samples']}")
        
        print("\n3. FEATURE ANALYSIS FOR MISCLASSIFIED SAMPLES")
        print("-" * 40)
        for feature, analysis in advanced_results['feature_analysis'].items():
            print(f"\n{feature}:")
            print("  Misclassified samples:")
            print(f"    Mean ± Std: {analysis['misclassified']['mean']:.2f} ± {analysis['misclassified']['std']:.2f}")
            print("  Correctly classified samples:")
            print(f"    Mean ± Std: {analysis['correct']['mean']:.2f} ± {analysis['correct']['std']:.2f}")
        
        print(f"\nVisualizations saved in: {advanced_results['visualization_directory']}")

def improved_print_analysis(analysis, advanced_analysis):
    """Print comprehensive analysis results"""
    print("\nDETAILED CLASSIFICATION ANALYSIS")
    print("=" * 50)
    
    print("\n1. MODEL PERFORMANCE")
    print("-" * 30)
    print(f"Ensemble Accuracy: {analysis['ensemble_performance']['accuracy']:.4f}")
    print(f"Ensemble F1 Score: {analysis['ensemble_performance']['f1_score']:.4f}")
    
    print("\n2. MODEL AGREEMENT")
    print("-" * 30)
    for key, value in analysis['prediction_analysis']['model_agreement'].items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    
    print("\n3. MISCLASSIFICATION PATTERNS")
    print("-" * 30)
    print(f"Total samples misclassified: {advanced_analysis['mistake_analysis']['total_mistakes']}")
    print(f"Common mistakes across all models: {advanced_analysis['mistake_analysis']['common_mistakes']}")
    print("\nUnique mistakes per model:")
    for model, count in advanced_analysis['mistake_analysis']['unique_mistakes'].items():
        print(f"  {model}: {count}")
    
    print("\n4. FEATURE PATTERNS IN MISCLASSIFIED SAMPLES")
    print("-" * 30)
    for feature in analysis['feature_importance'].index[:5]:  # Top 5 features
        mis_pattern = advanced_analysis['misclassification_patterns'][feature]
        cor_pattern = advanced_analysis['correct_patterns'][feature]
        print(f"\n{feature}:")
        print(f"  Misclassified - mean: {mis_pattern['mean']:.2f}, std: {mis_pattern['std']:.2f}")
        print(f"  Correct      - mean: {cor_pattern['mean']:.2f}, std: {cor_pattern['std']:.2f}")
    
    print("\n5. CONFIDENCE ANALYSIS")
    print("-" * 30)
    for model_name, results in analysis['prediction_analysis']['confidence_analysis'].items():
        print(f"\n{model_name}:")
        print(f"  Confidence gap: {results['mean_confidence_correct'] - results['mean_confidence_incorrect']:.4f}")
        print(f"  High confidence mistakes: {results['high_confidence_mistakes']}")


if __name__ == "__main__":
    classifier, training_results, test_results, analysis, X_test, y_test = train_compare_and_analyze()
    
    advanced_analyzer = WettabilityAdvancedAnalyzer(classifier)
    advanced_results = advanced_analyzer.add_advanced_analysis(X_test, y_test)
    
    print_analysis_results(analysis)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'training_results': training_results.to_dict(),
        'test_results': test_results.to_dict(),
        'analysis': analysis,
        'advanced_analysis': advanced_results
    }
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return obj

    def convert_dict_types(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = convert_dict_types(v)
            elif isinstance(v, (list, tuple)):
                d[k] = [convert_to_serializable(i) for i in v]
            else:
                d[k] = convert_to_serializable(v)
        return d

    results = convert_dict_types(results)
    
    with open(f'complete_analysis_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComplete results saved to: complete_analysis_results_{timestamp}.json")