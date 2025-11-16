use ndarray::{Array1, Array2, Axis, ArrayView1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use image::{DynamicImage, GenericImageView};

pub struct FuzzyCMeans {
    num_clusters: usize,
    fuzziness: f64,
    epsilon: f64,

    //data
    data: Array2<f64>, // X 


    //algo status 
    centers: Array2<f64>, // C
    memberships: Array2<f64>, // U
}

impl FuzzyCMeans {

    pub fn new(num_clusters: usize, fuzziness: f64, epsilon: f64, data: Array2<f64>) -> Self {
        let num_samples = data.nrows();
        let num_features = data.ncols();

        // Initialize centers and memberships
        let centers = Array2::<f64>::zeros((num_clusters, num_features));
        let memberships = Self::initialize_memberships(num_samples, num_clusters);

        let mut s = FuzzyCMeans {
            num_clusters,
            fuzziness,
            epsilon,
            data,
            centers,
            memberships,
        };
        s.update_centers();

        s 
         

    }

    pub fn run(&mut self, max_iterations: usize) {
        for _ in 0..max_iterations {
            let old_memberships = self.memberships.clone();

            self.update_centers();
            self.update_memberships();

            let max_delta = self.calculate_max_delta(&old_memberships);
            if max_delta < self.epsilon {
                break;
            }
        }
    }

    fn initialize_memberships(num_samples: usize, num_clusters: usize) -> Array2<f64> {
        let mut memberships = Array2::random((num_samples, num_clusters), Uniform::new(0.0, 1.0).unwrap());
        for mut row in memberships.axis_iter_mut(Axis(0)) {
            let row_sum: f64 = row.sum();
            if row_sum > 0.0 {
                row /= row_sum;
            }
        }
        memberships
    }

    //getter methods
    pub fn centers(&self) -> &Array2<f64> {
        &self.centers
    }
    pub fn memberships(&self) -> &Array2<f64> {
        &self.memberships
    }

    //update methods
    fn update_centers(&mut self) {
        // Update cluster centers based on current memberships
        let num_clusters = self.num_clusters;
        let num_points = self.data.nrows();
        let num_features = self.data.ncols();
        // C_j = sum_i (u_ij^m * x_i) / sum_i (u_ij^m)
        for j in 0..num_clusters {

            // Initialize numerator and denominator
            let mut numerator = Array1::<f64>::zeros(num_features);
            let mut denominator = 0.0;

            // Iterate over all data points
            for i in 0..num_points {

                let u_ij = self.memberships[[i, j]]; // membership coefficient
                let weight  = u_ij.powf(self.fuzziness);  // weight = u_ij^m

                denominator += weight; // sum_i (u_ij^m)

                let weighted_point = &self.data.row(i) * weight; // u_ij^m * x_i

                numerator = &numerator + &weighted_point; // sum_i (u_ij^m * x_i)

            }
            
            if denominator > 0.0 { // Avoid division by zero
                
                let new_center_j = &numerator / denominator; // C_j

                self.centers.row_mut(j).assign(&new_center_j); // Update center j
            }
        }
    }


    fn update_memberships(&mut self) {
        // Update membership coefficients based on current centers
        let num_clusters = self.num_clusters;
        let num_points = self.data.nrows();

        let exponent = 2.0 / (self.fuzziness - 1.0);
        let dist_epsilon = 1e-9;
        
    

        for i in 0..num_points {

            let points_i = self.data.row(i);

            let mut distances = Vec::with_capacity(num_clusters);
            let mut zero_distance_found_at_j : Option<usize> = None;

    

            for j in 0..num_clusters {
                let new_center_j = self.centers.row(j);
                let  dist = Self::calculate_distance(&points_i, &new_center_j);
                
                if dist < dist_epsilon {
                    zero_distance_found_at_j = Some(j);
                } else {
                    distances.push(dist);
                }
            }
            
            if let Some(j_zero) = zero_distance_found_at_j {
                let mut memberships_row = self.memberships.row_mut(i);
                memberships_row.fill(0.0);
                memberships_row[j_zero] = 1.0;
                continue;
            }

            for j in 0..num_clusters {
                let mut denominator_sum = 0.0;
                let dist_ij = distances[j];

                for k in 0..num_clusters {
                    let dist_ik = distances[k];
                    let ratio = dist_ij / dist_ik;
                    denominator_sum += ratio.powf(exponent);
                }

                let u_ij = 1.0 / denominator_sum;

                self.memberships[[i, j]] = u_ij;
            }


        }

        
    }

    fn calculate_max_delta(&self, old_memberships: &Array2<f64>) -> f64 {
        let diffs = &self.memberships - old_memberships;
        diffs.mapv(|x| x.abs()).fold(0.0, |a, b| a.max(*b))
    }

    fn calculate_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        // Euclidean distance
        let diff = a - b;
        diff.dot(&diff).sqrt()
        
    }

}

fn img_to_array(img: &DynamicImage) -> Array2<f64> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    let mut data = Array2::<f64>::zeros(((width * height) as usize, 3));

    for (x, y, pixel) in rgb_img.enumerate_pixels() {
        let idx = (y * width + x) as usize;
        data[[idx, 0]] = pixel[0] as f64;
        data[[idx, 1]] = pixel[1] as f64;
        data[[idx, 2]] = pixel[2] as f64;
    }

    data
}
fn main() {
       use std::fs;
    use std::path::Path;

    println!("Fuzzy C-Means batch segmentation");

    let img_path = "milky-way.jpg";
    let img = image::open(img_path).expect("Failed to open image");
    let (width, height) = img.dimensions();
    let data = img_to_array(&img); // correction de la coquille

    println!("Image loaded: {}x{}", width, height);

    // Crée le dossier de sortie
    let out_dir = Path::new("fcm_outputs");
    fs::create_dir_all(out_dir).expect("Failed to create output directory");

    // 5 jeux de paramètres différents (k, fuzziness m, epsilon, max_iter)
    let params: Vec<(usize, f64, f64, usize)> = vec![
        (3, 1.5, 0.02, 100),
        (5, 2.0, 0.01, 100),
        (8, 2.5, 0.005, 150),
        (10, 2.0, 0.01, 120),
        (12, 1.8, 0.02, 80),
    ];

    for (idx, (k, m, eps, max_iter)) in params.iter().enumerate() {
        println!("Run {}: k={}, m={}, eps={}, max_iter={}", idx + 1, k, m, eps, max_iter);

        let mut fcm = FuzzyCMeans::new(*k, *m, *eps, data.clone());
        fcm.run(*max_iter);

        // Construire un nom de fichier sans points problématiques
        let m_str = format!("{}", m).replace('.', "p");
        let eps_str = format!("{}", eps).replace('.', "p");
        let filename = format!("seg_{:02}_k{}_m{}_eps{}_it{}.png", idx + 1, k, m_str, eps_str, max_iter);

        let out_path = out_dir.join(&filename);
        let out_path_str = out_path.to_str().expect("Invalid output path");

        save_segmented_image(&fcm, width, height, out_path_str);
        println!("Saved: {}", out_path_str);
    }

    println!("Batch segmentation completed. Outputs in {:?}", out_dir);
}



fn get_max_membership_index(membership_row: &ArrayView1<f64>) -> usize {
    membership_row
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}


fn save_segmented_image(
    fcm: &FuzzyCMeans,
    width: u32,
    height: u32,
    output_path: &str,
) {

    let mut segmented_img = image::RgbImage::new(width, height);
    let centers = fcm.centers();
    let memberships = fcm.memberships();

    for (i, pixel_memberships) in memberships.rows().into_iter().enumerate() {
        let cluster_index = get_max_membership_index(&pixel_memberships);

        let center_color = centers.row(cluster_index);
        let r = center_color[0].round().clamp(0.0, 255.0) as u8;
        let g = center_color[1].round().clamp(0.0, 255.0) as u8;
        let b = center_color[2].round().clamp(0.0, 255.0) as u8;

        let x = (i % (width as usize)) as u32;
        let y = (i / (width as usize)) as u32;
        segmented_img.put_pixel(x, y, image::Rgb([r, g, b]));
    }

    segmented_img.save(output_path).expect("Failed to save segmented image");
    println!("Image segmentée sauvegardée sous : {}", output_path);
}