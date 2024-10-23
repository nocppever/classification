import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from datetime import datetime, timedelta

class WettabilityDataGenerator:
    def __init__(self):
        
        self.materials = {
            'Verre': {
                'surface_energy_range': (70, 80),  # mN/m
                'thermal_conductivity_range': (0.8, 1.2),  # W/(m·K)
                'youngs_modulus_range': (65, 75),  # GPa
                'density_range': (2.4, 2.8),  # g/cm³
                'base_contact_angle_range': (20, 40)  # degres
            },
            'PTFE': {
                'surface_energy_range': (18, 22),
                'thermal_conductivity_range': (0.23, 0.27),
                'youngs_modulus_range': (0.4, 0.6),
                'density_range': (2.1, 2.2),
                'base_contact_angle_range': (108, 122)
            },
            'Inox': {
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
            'Wafer': {
                'surface_energy_range': (45, 55),
                'thermal_conductivity_range': (130, 150),
                'youngs_modulus_range': (130, 170),
                'density_range': (2.32, 2.33),
                'base_contact_angle_range': (35, 45)
            }
        }

        self.surface_treatments = {
            'non traiter': {
                'angle_modifier': (0, 5),
                'roughness_range': (0.1, 0.3)
            },
            'Plasma': {
                'angle_modifier': (-40, -20),
                'roughness_range': (0.05, 0.2)
            },
            'Silane': {
                'angle_modifier': (30, 45),
                'roughness_range': (0.2, 0.5)
            },
            'Fluorination': {
                'angle_modifier': (60, 80),
                'roughness_range': (0.3, 0.8)
            },
            'UV_Ozone': {
                'angle_modifier': (-30, -15),
                'roughness_range': (0.1, 0.3)
            }
        }

        self.environmental_conditions = {
            'temperature_range': (20, 25),  # °C
            'humidity_range': (40, 60),     # %
            'pressure_range': (995, 1015)   # hPa
        }

    def get_truncated_normal(self, mean, sd, low, high):
        """normal distribution"""
        return truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd)

    def generate_data(self, n_samples=1000, include_noise=True, include_environmental=True):
        """
        Generation synthetique de dataset pour mouillabilite.
        """
        data = []
        
        for _ in range(n_samples):
            
            material = np.random.choice(list(self.materials.keys()))
            treatment = np.random.choice(list(self.surface_treatments.keys()))
            
            material_props = self.materials[material]
            treatment_props = self.surface_treatments[treatment]
            
            
            sample = {
                'material': material,
                'surface_treatment': treatment,
                'surface_energy': np.random.uniform(*material_props['surface_energy_range']),
                'thermal_conductivity': np.random.uniform(*material_props['thermal_conductivity_range']),
                'youngs_modulus': np.random.uniform(*material_props['youngs_modulus_range']),
                'density': np.random.uniform(*material_props['density_range']),
                'roughness_ra': np.random.uniform(*treatment_props['roughness_range'])
            }
            
            base_porosity = 0.1 if material in ['Glass', 'Silicon_Wafer'] else 0.2
            treatment_factor = 1.5 if treatment in ['Plasma', 'UV_Ozone'] else 1.0
            sample['porosity'] = min(0.6, base_porosity * treatment_factor * np.random.uniform(0.8, 1.2))
            
            # Calculer contact angle
            base_angle = np.random.uniform(*material_props['base_contact_angle_range'])
            angle_modifier = np.random.uniform(*treatment_props['angle_modifier'])
            contact_angle = base_angle + angle_modifier
            
            # Bruit Gaussien
            if include_noise:
                contact_angle += np.random.normal(0, 2)  
            
            sample['contact_angle'] = max(0, min(180, contact_angle)) 
            
            # Conditions environnementales
            if include_environmental:
                sample.update({
                    'temperature': np.random.uniform(*self.environmental_conditions['temperature_range']),
                    'humidity': np.random.uniform(*self.environmental_conditions['humidity_range']),
                    'pressure': np.random.uniform(*self.environmental_conditions['pressure_range'])
                })
            
            
            surface_energy_factor = 1 + (sample['surface_energy'] - 50) / 200
            sample['contact_angle'] *= surface_energy_factor
            
            
            sample['wettability_class'] = (
                'Superhydrophilic' if sample['contact_angle'] < 10 else
                'Hydrophilic' if sample['contact_angle'] < 90 else
                'Hydrophobic' if sample['contact_angle'] < 150 else
                'Superhydrophobic'
            )
            
            data.append(sample)
        
        return pd.DataFrame(data)

def create_dataset(n_samples=1000, save_to_csv=False, include_environmental=True):
    """Create a dataset with the enhanced generator"""
    generator = WettabilityDataGenerator()
    df = generator.generate_data(n_samples, include_environmental=include_environmental)
    
    if save_to_csv:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'synthetic_wettability_data_{timestamp}.csv'
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    
    return df

if __name__ == "__main__":
    # Generer dataset
    df = create_dataset(n_samples=2000, save_to_csv=True, include_environmental=True)
    
    
    print("\nResume Dataset:")
    print("-" * 50)
    print(f"Total echs: {len(df)}")
    
    print("\nDistribution des materiaux:")
    print(df['material'].value_counts())
    
    print("\nDistribution des traitements de surface:")
    print(df['surface_treatment'].value_counts())
    
    print("\nDistribution des classes de mouillabilite:")
    print(df['wettability_class'].value_counts())
    
    print("\nAngle de Contact:")
    print(df.groupby('material')['contact_angle'].describe())