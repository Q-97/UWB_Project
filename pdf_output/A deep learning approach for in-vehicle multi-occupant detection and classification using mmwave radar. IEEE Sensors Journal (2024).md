# A Deep Learning Approach for In-Vehicle Multi-Occupant Detection and Classification Using mmWave Radar

Jayson P. Van Marter , Student Member, IEEE, Anand G. Dabak , Fellow, IEEE, Anil Varghese Mani, Member, IEEE, Sandeep Rao , Senior Member, IEEE, and Murat Torlak , Senior Member, IEEE

Abstract—Due to several benefits including a wide field of view, fine resolution, and low cost, millimeter-wave (mmWave) radars are of high interest for in-vehicle sensing tasks, including occupant detection and classification. While general presence detection, which identifies any living presence across all seats, is very accurate using model-based methods, limited angular resolution, multipath reflections, and ambient reflections impede localized seat-byseat detection and classification of occupants. In this article, we propose a novel deep learning solution using 3-D point clouds obtained from an mmWave radar mounted in-cabin.

<!-- image-->

By focusing on sparse 3-D point clouds rather than fully populated range-angle heatmaps, we obtain a 54.1% reduction in computational complexity and a 94.7% reduction in data storage requirements while preserving estimated velocity per point. Furthermore, our method addresses challenges due to ambient and multipath reflections in the vehicle by constraining the spatial focus of our model. Evaluations demonstrate 95.6% accuracy for localized detection and 88.7% accuracy for classifying the occupant as adult, child, or baby when testing on participants unseen during training and validation.

Index Terms— Child presence detection (CPD), deep learning, in-cabin, in-vehicle, millimeter-wave (mmWave) radar, radar sensing, target classification, target detection.

## I. INTRODUCTION

N RECENT years, children have commonly occupied I the back seats of vehicles to prevent injury or death from overpowered front seat airbags. However, there has been a connected uptick in the number of child deaths due to heatstrokes from being forgotten in the car [1], [2]. In response, the HOT CARS Act was proposed in U.S. Congress in 2019, which would require new vehicles to be equipped with rear seat reminder technologies (RSRTs) [3]. Similarly, the European New Car Assessment Program (Euro NCAP) has rewarded safety points to vehicles that can perform child presence detection (CPD) since 2022 [4]. In addition to CPD, new cars are currently required by the National Highway Traffic Safety Administration (NHTSA) to provide seat belt reminder (SBR) warnings when the driver is not wearing a seat belt. In 2023, the NHTSA has proposed to expand SBR warnings to include all vehicle passengers [5]. Incumbent technologies that may perform these tasks, such as weight or pressure sensors, would require installing a sensor in each seat, increasing costs. Moreover, these devices are unable to differentiate living from nonliving occupants. Thus, there are opportunities for new low-cost technologies such as millimeter-wave (mmWave) radar to be installed as a single sensor that performs occupant detection across all seats.

Digital Object Identifier 10.1109/JSEN.2024.3450432

Then, by extending capabilities to occupant classification, these sensors can enable enhanced features such as smart airbags, smart climate control, and advanced seating positions.

Existing occupant sensing techniques have been based on in-seat methods, which require installing a sensor for each seat, or noncontact methods using radio frequency (RF), infrared, or vision sensors [6]. In-seat methods include those based on weight sensors [7], [8], pressure sensors [9], and capacitive sensors [10], [11], [12], [13]. The primary limitation of weight and pressure sensors is their inability to differentiate between living and non-living targets in the seat. While capacitive sensors can mitigate this limitation, they are sensitive to distance to the occupant and electromagnetic interference. Moreover, they still require at least one sensor to be installed per seat, increasing costs. Noncontact methods based on pyroelectric infrared sensors (PIRs) [14], [15] and vision sensors [16], [17], [18], [19] enable a wider sensing area that covers multiple seats. However, environment temperature can degrade the reliability of PIR sensors in practice. Vision sensors, such as cameras, can facilitate high accuracy using data-driven image processing techniques. However, camera sensors fundamentally require line-of-sight vision of the target, limiting their capabilities in poor lighting conditions or cases where a passenger may not be visible due to a blanket or other line-ofsight blockage. Vision sensors are also not privacy-preserving.

RF-based solutions have been investigated due to several notable benefits. RF-based systems are robust to environment conditions, including smoke, dust, fog, and poor illumination. They also have notable penetrative capabilities and a small form factor, enabling them to measure through objects such as blankets and be installed in noninvasive locations and chassis. Finally, RF-based systems are low power, low cost, and privacy-preserving compared to alternative visionbased systems. Existing solutions that leverage RF signals primarily include those based on WiFi, ultra-wide band (UWB), and mmWave frequency-modulated continuous wave (FMCW) radar [6], [20]. WiFi and UWB solutions seek to build on existing systems in cars, i.e., the WiFi connectivity interface and UWB keyless infrastructure, respectively [21], [22]. Impulse radio UWB (IR-UWB) radar has recently been investigated due to its passive ranging, i.e., device-to-target, capabilities, compared to the device-to-device active ranging of traditional UWB signaling. These works primarily rely on co-located transmitter and receiver antennas to operate as a radar using short pulses. With the high resolution of ranging offered by UWB, ranges to different seats can be utilized for localized detection of occupants, and measurements over time can be used to obtain velocity information [23], [24], [25], [26], [27], [28], [29]. In comparison, common mmWave FMCW radars offer reliable directionality using multi-input multi-output (MIMO) antenna topologies in addition to highresolution velocity and range information [30].

Approaches to in-vehicle occupant sensing using mmWave FMCW radar have been primarily based on vital signs statistics [1], [31], [32], [33], [34], [35] or point cloud statistics [36], [37]. While these model-based methods can perform CPD with nearly 100% accuracy, localized detection per seat has been more challenging due to limited radar angular resolution, multipath reflections, and ambient reflections. Moreover, identifying reliable features for model-based classification of occupants is challenging and has not been explored in prior literature.

The contributions of this work can be outlined as follows.

1) To overcome the limitations of model-based methods and automate relevant feature extraction, we propose a novel deep learning framework for occupant sensing tasks using mmWave FMCW radar. Our framework builds on recent advances in data-driven point cloud analysis and human sensing with radar to extract spatial and temporal features [38], [39], [40], [41], [42].

2) We develop a multipass constant false alarm rate (CFAR) thresholding and zoom-in beamforming approach to systematically obtain points of interest with high resolution. Our preprocessing approach to obtain point cloud information achieves a complexity reduction of 54.1% and a data storage requirements reduction of 94.7% compared to the fully populated 3-D heatmap while maintaining velocity information and increasing effective angular resolution.

3) We evaluate the performance of our approach by testing on groups of participants unseen by the model during training and validation. Evaluations are made for localized per-seat detection, occupant classification as baby, child, or adult, and smart CPD, which detects if one or more children are in the vehicle alone or if an adult is present. Across these tasks, we achieve considerable accuracies of 95.6% for localized detection, 88.7% for classification of new occupants, and 88.4% for smart CPD.

The rest of the article is organized as follows. In Section II, we discuss related works and contextualize our contributions. Section III outlines the mmWave FMCW radar signal model and our method to obtain point cloud data through multipass CFAR thresholding and zoom-in beamforming. In Section IV, we describe the deep learning network design to extract features and perform occupant sensing tasks using the point cloud data. Section V explains our collected dataset used for evaluation. In Section VI, we evaluate the complexity and performance of our approach across different occupant sensing tasks and metrics. Finally, Section VII concludes the article.

## II. RELATED WORKS

Solutions for in-vehicle occupant presence detection and localized seat-by-seat detection have been high-interest research topics in recent years, with detailed surveys available in [6] and [20]. Radar-based solutions have been primarily based on either IR-UWB radar or mmWave FMCW radar. For IR-UWB radar, methods can be categorized as model-based, using vital signs statistics, or data-driven, using machine learning with measured channel impulse responses (CIRs) over time. Similarly, for mmWave FMCW radar, approaches can be categorized as those based on vital signs statistics, point cloud statistics, or, most recently, machine learning using range-angle heatmaps. In the following, we discuss radarbased works employing each of these approaches.

## A. IR-UWB Radar

In this section, we survey works that utilize IR-UWB radar for in-cabin occupant presence detection and localized seatby-seat detection, categorized by their approach.

1) Vital Signs Methods: Vital signs detection and estimation has a wide array of applications and is a problem that has gained a lot of attention in recent years [43]. For in-vehicle localized detection, small movements due to breathing and heartbeat can be tracked and used to differentiate living from nonliving targets. Works that directly employ this idea include [23], [24]. Möderl [23] use a single IR-UWB radar and a single-point-target model. Following the model, they detect breathing motions by employing an estimator-correlator on the covariance of the channel delay profile and power spectrum of the expected chest motion. Results with different signal-to-noise ratios (SNR) are demonstrated using a single target in a vehicle, and performance is improved compared to a Fourier-transform-based detector. Fioranelli et al. [24] utilize two IR-UWB radars with various antenna placements and evaluate up to two occupied seats. For their approach, they first extract range-Doppler maps from CIRs received over time and across multiple IR-UWB receivers. From these maps, they employ 2-D CFAR to obtain range-Doppler bins that may contain vital signs. Estimated ranges and breathing frequencies across all receiver antennas are utilized in a cost function, from which the final target positions and breathing frequencies are estimated. Results demonstrate a localization root mean square error of approximately 16 cm for both singleoccupant and two-occupant scenarios in a vehicle. Different radar network topologies are also explored in simulation, demonstrating improvement with bistatic geometries.

2) Machine Learning Methods: In [25], features of the received CIR from a single IR-UWB radar, including mean, variance, maximum value, etc., are used with the neighborhood component analysis algorithm to assign feature importance. Then, ensemble learning with a decision tree is used to determine seat occupancy. Results demonstrate a detection accuracy of more than 90% across five seats. Later, in [26], a deep neural network (DNN) is proposed and compared to a support vector machine (SVM) and the decision-tree-based method. While the decision-tree-based method previously required manual feature selection, the DNN works with the CIR directly, learning relevant features from the data. For their approach, detection accuracy reaches 99% across five seats. Then, in [27], two radars are used instead of one to obtain distance-time maps for a CNN-based solution, with detection accuracy reaching 93%. In [28], features are extracted from CIR measurements taken over time from a single IR-UWB radar, categorized as two types: spatial–temporal-circulated gray-level co-occurrence matrix (STC-GLCM) and physiological. Comparisons are made across different feature extraction methods and learning algorithms, including an SVM, stochastic gradient descent (SGD), and random forest methods. Using SGD, accuracy reaches 97.1% for counting up to five occupants inside a vehicle. Classification accuracy for empty, lying, or sitting reaches 99.0%. Finally, in [29], two IR-UWB radars are used to determine occupancy by taking CIR measurements over time and feeding into a network based on the ResNet architecture [44]. Evaluations are made to determine if the entire car is empty or occupied, and it is demonstrated that with a high enough SNR, accuracy approaches 100% for different activities, including breathing, talking, and moving.

## B. mmWave FMCW Radar

We next focus on works that use mmWave FMCW radar for in-vehicle sensing solutions toward occupant presence detection and localized seat-by-seat detection. While IR-UWB radar may be able to leverage existing UWB keyless infrastructure, common mmWave FMCW radars offer the key benefit of directionality information using MIMO antenna topologies. This benefit comes with operation at millimeter-level wavelengths, which enables compact antenna placements without loss of directionality resolution.

1) Vital Signs Methods: Similar to IR-UWB radar, using the high sensitivity and resolution of mmWave FMCW radars, small movements from heartbeat and respiratory cycles can be measured and tracked [31], [43]. Following the periodicity of these signals, living targets can be differentiated from nonliving targets. Among the earliest works that employ this principle is [1], in which a 24 GHz continuous-wave radar is used to measure periodic breathing by cross correlation. With their approach, they demonstrate the ability to detect a baby doll after 8–10 s of measurements. Following up this work in [32], a 24 GHz FMCW radar is used to determine the localized presence of a person across three seats in an anechoic chamber. To detect vital signs, static clutter removal and a bandpass filter are used. Similarly, in [33], a 60 GHz FMCW radar is used to detect heartbeat and respiratory responses in their expected frequency ranges of the time-frequency spectrum across two seats. Their approach is extended in [34] where occupancy detection performance is evaluated with linear discriminant analysis (LDA) and maximum likelihood estimation (MLE). While accuracy exceeds 98% following MLE, the results are only evaluated in the two back corner seats of the vehicle. Lastly, the work in [35] uses a 77 GHz FMCW radar and cross correlation to detect the presence of respiratory cycles corresponding to an occupant. Results across adults and baby dolls demonstrate 100% detection accuracy. However, localized detection across multiple seats is not evaluated. Among these works, only [32], [33], and [34] evaluate localized detection across seats. Then, only [34] statistically quantifies performance. These works do not evaluate localized detection performance across both the front and back seats of a vehicle, and none take measurements with children, who are often active and moving, increasing the difficulty of measuring vital signs accurately.

2) Point Cloud Methods: To obtain a 3-D point cloud in space from FMCW radar measurements, static clutter removal and CFAR thresholding are used to obtain the most significant reflected points [45], [46], [47]. Then, range and angle information are converted to Cartesian coordinates in space. Using the detected point cloud, features such as the number of detected points, volume of the point cloud, or average SNR can be used to make a decision. The work in [36] uses a 77 GHz FMCW radar and thresholds based on the number of point cloud clusters to determine localized occupant presence in the rear seats of a vehicle. Similarly, the work in [37] uses a 77 GHz FMCW radar and thresholds based on the standard deviation and density of points in each defined seat zone. They demonstrate 96% and 90% accuracy when determining the number of occupants in four- and five-seater vehicles, respectively, and 86.1% accuracy when determining the seat locations of all occupants in a four-seater vehicle.

3) Machine Learning Methods: Thus far, methods based on machine learning have been based on fully populated rangeangle heatmaps, which provide reflected power information across range and angle. The work in [48] uses a 77 GHz FMCW radar to evaluate random forest, K -nearest neighbors (KNNs), and SVM methods for localized occupancy detection with 2-D range-azimuth heatmaps. Cross-validation with five folds is used, and the SVM approach has the highest performance at 97% accuracy for determining the locations of all occupants in the vehicle. However, evaluations are made on the same participants that appeared in the training set. In [49], a large-array $2 0 \times 2 0$ multiple-input multipleoutput (MIMO) FMCW radar operating at 60 GHz is used with a deep learning model based on a 3-D convolutional neural network (CNN) and a long short-term memory (LSTM) network, a type of recurrent neural network (RNN) [50]. Evaluations are made on participants unseen in the training set with 95% accuracy for determining occupant presence, 89% accuracy for localizing all occupants across five seats, and 74% accuracy for classification of all seats as empty, child, or adult. While this work demonstrates considerable accuracy for in-vehicle occupant sensing tasks, reliance on a fully populated 3-D heatmap of size $2 9 \times 2 9 \times 2 4$ for azimuth, elevation, and range dimensions, respectively, implies considerable data storage requirements. Furthermore, only two classes of occupants are evaluated for classification: child and adult. In [51], a 60 GHz radar is used to obtain 3-D heatmaps in range, azimuth, and elevation dimensions. For localization and classification of occupants, a 3-D CNN is used which operates per seat of the vehicle. The model shows good generalization performance in using a single DNN across five seats. Evaluations on unseen occupants demonstrate 95.5% accuracy for occupant localization and 85.1% accuracy for determining if an occupant is a baby, child, or adult. Rather than range-angle heatmaps, in this work, we consider sparse 3-D point cloud data for occupant sensing tasks to reduce redundancies, computational complexity, and data storage requirements. We then evaluate both occupant localization and classification of occupants across three passenger classes: baby, child, and adult.

## III. SYSTEM MODEL

In this section, we outline the FMCW radar signal model and preprocessing to obtain point cloud data with estimated SNR, range, and velocity for each point. A high-level flowchart of the preprocessing is provided in Fig. 1. This preprocessing is completed for each radar frame consisting of L FMCW chirps. A transmitted FMCW chirp signal can be expressed in time as

$$
s ( t ) = \exp { \left[ j 2 \pi \left( f _ { 0 } t + \frac { K } { 2 } t ^ { 2 } \right) \right] } , \quad 0 \leq t \leq T\tag{1}
$$

<!-- image-->  
Fig. 1. Radar preprocessing flowchart. The $x \cdot , y \cdot ,$ and z-coordinates are obtained through a coordinate transform from the range, azimuth, and elevation.

where $f _ { 0 }$ is the chirp start frequency at time $t ~ = ~ 0$ and $\begin{array} { r } { K \ = \ B / T } \end{array}$ is the chirp slope for total bandwidth B and duration T . The linear modulated frequency can be expressed as $f ( t ) = f _ { 0 } + K t$ . Extending this model to include multiple chirps transmitted in a frame, we have

$$
s _ { T } ( t ) = \sum _ { l = 0 } ^ { L - 1 } s ( t - l T )\tag{2}
$$

where l is the chirp index. For a point target, each reflected chirp is observed as

$$
r _ { p } ( t ) = \alpha \exp \left[ j 2 \pi \left( f _ { 0 } ( t - \tau _ { p } ) + \frac { K } { 2 } ( t - \tau _ { p } ) ^ { 2 } \right) \right]\tag{3}
$$

where $p$ is the transmit–receive antenna pair in a virtual array [52], [53], α is the observed attenuation due to path loss and target reflectivity, and $\tau _ { p }$ is the two-way time of flight (ToF) delay. Mixing the received signal, sampling, and expanding phase terms, the beat signal is obtained as

$$
z _ { n , p } \left( \tau _ { p } \right) = \alpha \exp \left[ - j 2 \pi \left( f _ { 0 } ^ { \mathrm { ( I F ) } } \tau _ { p } + K n T _ { s } \tau _ { p } \right) \right]\tag{4}
$$

where $f _ { 0 } ^ { \mathrm { ( I F ) } }$ is the intermediate frequency (IF), $n = 0 , \ldots ,$ $N - 1 , \ \bar { T _ { s } }$ is the sampling interval, and the negligible residual video phase (RVP) term π $K \tau _ { p } ^ { 2 }$ is ignored [54]. Expanding the beat signal across multiple chirps, we have

$$
\begin{array} { r l } {  { z _ { n , p , l } ( \tau _ { 0 } , \phi , \theta , f _ { D } ) } \quad } & { } \\ & { = \alpha \exp \big [ - j 2 \pi \big ( f _ { 0 } ^ { \mathrm { ( I F ) } } \tau _ { 0 } + K n T _ { s } \tau _ { 0 } + f _ { D } l T } \\ & { \hphantom { = \exp x } + ( x _ { p } / \lambda ) \sin ( \phi ) + ( y _ { p } / \lambda ) \sin ( \theta ) \big ) \big ] } \end{array}\tag{5}
$$

where $\tau _ { 0 }$ is the ToF to the $p = 0$ virtual antenna element, $x _ { p }$ is the location of the pth virtual antenna in the x-axis, $y _ { p }$ is the location of the pth virtual antenna in the y-axis, φ is the direction of arrival (DoA) in azimuth, θ is the DoA in elevation, $f _ { D }$ is the Doppler frequency shift, and λ is the carrier wavelength. The ToF of the received beat signal across antennas and chirps can then be expressed as

$$
\tau _ { p , l } = \frac { 2 R + 2 l V T + x _ { p } \sin ( \phi ) + y _ { p } \sin ( \theta ) } { c }\tag{6}
$$

where R is the two-way distance to the target and V is the target velocity.

To aid in further analysis of the beat signal, we adopt the matrix notation

$$
\mathbf { z } _ { p , l } = \bigl [ z _ { 0 , p , l } , z _ { 1 , p , l } , . . . , z _ { N - 1 , p , l } \bigr ] ^ { \intercal }\tag{7}
$$

where ⊺ represents the transpose operation. To extract power density across the range from the beat signal, we first take a discrete Fourier transform (DFT) as

$$
\begin{array} { r l } & { \mathbf { Z } _ { p , l } = \Phi _ { K } \mathbf { z } _ { p , l } } \\ & { \qquad = \bigl [ Z _ { 0 , p , l } , Z _ { 1 , p , l } , \ldots , Z _ { K - 1 , p , l } \bigr ] ^ { \intercal } } \end{array}\tag{8}
$$

where $\Phi _ { K }$ is the DFT matrix of size K . In practice, this DFT is implemented as a fast Fourier transform (FFT), and the array size in range is reduced after the FFT to correspond to ranges of interest. The range at each index k can be expressed as $R _ { k } ~ = ~ ( k c / 2 K B )$ . Since our goal is to observe living targets, we seek to only measure moving points present due to occupant movement or vital signs. To mitigate reflections from still targets such as the vehicle interior and car body, we employ static clutter removal as

$$
\mathbf { Z } _ { p , l } ^ { s } = \mathbf { Z } _ { p , l } - \frac { 1 } { L } \sum _ { l = 0 } ^ { L - 1 } \mathbf { Z } _ { p , l }\tag{9}
$$

where l is the chirp index. To obtain observations across azimuth and elevation angles for each range, we adjust the dimensions of interest to form $P \times L$ matrices across virtual antennas and chirps, respectively, as

$$
\mathbf { A } _ { k } ^ { s } = \left[ \mathbf { Z } _ { k , 0 } ^ { s } , \mathbf { Z } _ { k , 1 } ^ { s } , \ldots , \mathbf { Z } _ { k , L - 1 } ^ { s } \right]\tag{10}
$$

where $\mathbf { Z } _ { k , l } ^ { s } = [ Z _ { k , 0 , l } ^ { s } , Z _ { k , 1 , l } ^ { s } , \ldots , Z _ { k , P - 1 , l } ^ { s } ] ^ { \intercal }$ and $P$ is the number of virtual antennas. To obtain DoA information, we use Capon beamforming rather than FFTs to improve angular resolution [55]. For Capon beamforming, a covariance matrix estimate across antennas can be obtained as

$$
\mathbf { R } _ { k } = \mathbf { A } _ { k } ^ { s } \left( \mathbf { A } _ { k } ^ { s } \right) ^ { H } + \beta \mathbf { I }\tag{11}
$$

where H represents conjugate transpose, $\beta$ is the diagonal loading parameter, and I is the identity matrix of size $P \times P$ The steering vectors for each azimuth angle $\phi _ { i }$ and elevation angle $\theta _ { m }$ of interest are defined as

$$
\begin{array} { c } { \displaystyle \mathbf { a } _ { i , m } = [ \exp ( - j 2 \pi \frac { 1 } { \lambda } ( x _ { 0 } \sin ( \phi _ { i } ) + y _ { 0 } \sin ( \theta _ { m } ) ) ) ] , } \\ { \displaystyle \exp ( - j 2 \pi \frac { 1 } { \lambda } ( x _ { 1 } \sin ( \phi _ { i } ) + y _ { 1 } \sin ( \theta _ { m } ) ) ) , \ldots , } \\ { \displaystyle \exp ( - j 2 \pi \frac { 1 } { \lambda } ( x _ { P - 1 } \sin ( \phi _ { i } ) + y _ { P - 1 } \sin ( \theta _ { m } ) ) ) ] ^ { \intercal } . } \end{array}\tag{12}
$$

The range-angle heatmap can then be calculated as

$$
H _ { k , i , m } = \frac { 1 } { \mathbf { a } _ { i , m } ^ { H } \mathbf { R } _ { k } ^ { - 1 } \mathbf { a } _ { i , m } }\tag{13}
$$

where $H _ { k , i , m }$ is the observed power at range index $k ,$ azimuth angle index i, and elevation angle index m. To obtain point cloud data from this heatmap, multipass CFAR thresholding is used as described in Fig. 2. Detailed algorithm steps for our multipass CFAR detection approach with zoom-in Capon beamforming are provided in Appendix I. In the first pass, significant heatmap points are found by comparing the power of the point of interest to the power of adjacent range values for given azimuth and elevation angles. Likewise, in the second pass, detected points are found by comparing the power of the point of interest to the power of adjacent azimuth angles for a given range that was detected in the first pass. This two-pass approach ensures diverse noise estimates from both adjacent range and angle bins for detection.

<!-- image-->

2. CFAR Second Pass: Azimuth Samples  
<!-- image-->

3. Zoom-In Capon Beamforming and CFAR Pass  
<!-- image-->  
Fig. 2. Multipass CFAR thresholding with zoom-in Capon beamforming. For the first two passes in range and azimuth, the average powers, µL and µR, around the sample under test $H _ { k , i , m }$ are used to estimate the noise, $\dot { W } _ { k , i , m } ^ { ( R ) } \circ \mathrm { r } \ W _ { k , i , m } ^ { ( \phi ) } .$ The noise estimate is then used to determine if the point is strong enough to be detected. For the zoom-in Capon beamforming and CFAR pass, the powers of the strongest Gmax and weakest $G _ { \mathrm { m i n } }$ points are used to form a threshold for detection.

Zoom-in Capon beamforming is implemented to obtain additional significant points near those detected from the first two CFAR passes. This approach enables high-resolution scanning in range and azimuth dimensions without exhaustively computing the heatmap for all possible points. Zoom-in Capon beamforming follows the same steps as (12) and (13) for a limited area of interest determined by selected zoom-in steering vector angles $\phi _ { u }$ and $\theta _ { v }$ . Then, a CFAR thresholding pass is completed across the zoom-in angles to obtain the final detected points indicated by $J _ { k , i , m , u , v } = 1$ in Algorithm 2 of Appendix I. The SNR of each point is estimated as

$$
S _ { k , i , m , u , v } = \frac { G _ { k , i , m , u , v } } { W _ { k , i , m } ^ { ( R ) } }\tag{14}
$$

where $G _ { k , i , m , u , v }$ is the full heatmap with zoom-in samples and $W _ { k , i , m } ^ { ( R ) }$ is the noise estimate obtained from the first CFAR pass across range samples.

Finally, for each detected point, the velocity spectrum is calculated as

$$
\mathbf { V } _ { k , i , m , u , v } = \Phi _ { D } \left[ \left( \frac { \mathbf { R } _ { k } ^ { - 1 } \mathbf { c } _ { i , m , u , v } } { \mathbf { c } _ { i , m , u , v } ^ { H } \mathbf { R } _ { k } ^ { - 1 } \mathbf { c } _ { i , m , u , v } } \right) ^ { H } \mathbf { A } _ { k } ^ { s } \right] ^ { \mathsf { T } }\tag{15}
$$

<!-- image-->

(a)  
<!-- image-->  
Fig. 3. (a) Single frame of detected point cloud points with ambient and multipath reflections in defined seat areas other than occupied ones. (b) Superposition of point cloud points across five sequential frames, including the frame of (a).

where $\mathbf { c } _ { i , m , u , v } \ = \ \mathbf { a } _ { i , m } \odot \mathbf { b } _ { u , v } , \ \mathbf { b } _ { u , v }$ is the zoom-in steering vector for the uth azimuth angle and vth elevation angle, ⊙ indicates the Hadamard product, and D is the DFT size. The velocity indices swept are based on the DFT size as $V _ { d \mathrm { ~ } } = \ ( d \lambda / 2 D T )$ where $d = 0 , \dots , D - 1$ indicates the velocity spectrum index. The velocity of each detected point is estimated as the $V _ { d }$ corresponding to the maximum of the velocity spectrum.

## IV. IN-CABIN OCCUPANT SENSING MODEL

While methods for general presence detection have shown considerable accuracy using radar for in-vehicle applications [34], [35], localized occupancy detection and classification of occupants remains challenging [37], [48], [49]. These challenges arise due to limited angular resolution, multipath reflections, and ambient reflections. Specifically, angular resolution is limited by the number and spacing of radar antenna elements. Multipath reflections appear from target signals reflecting off both the target and the car interior before returning to the radar. Similarly, ambient reflections appear from the car interior. While the car body is typically unseen in the heatmap due to static clutter removal, its reflections may appear if the occupant moves enough to shift the car body. Ambient and multipath reflections are visualized in Fig. 3 for measurements taken in a vehicle with two occupants.

An additional challenge of occupant sensing using the radar point cloud is the definition of relevant features and thresholds. Several features, such as the number of point cloud clusters, the density of each point cloud, the number of points, and the average SNR of the points, may be suitable, but it is difficult to relate all possible features for a robust decision. Moreover, point cloud data is irregular and requires a sensing model that is agnostic to the input sequence order.

To overcome these challenges and obtain a robust algorithm for in-vehicle occupant sensing tasks, we develop a DNN, which learns features from the data and utilizes the attention mechanism to overcome the irregularity of point cloud data [38], [42]. Our developed model employs learnings from recent advances in both general point cloud processing [40], [41], [42] and radar sensing in the related task of human pose estimation [38], [39], [56]. Toward mitigation of the effects of multipath and ambient reflections, we additionally constrain the problem by defining 3-D seat zones as shown in Fig. 3. If the point cloud points fall outside of these zones, then they are not detected, thus, the network is forced to focus on points directly reflected from the subject in the seat.

As measurements are taken across time, a single time unit of interest is a frame consisting of a fixed number of chirps. However, the point cloud of a single frame is very sparse, often consisting of less than 100 points that correspond to reflections from an occupant [38]. To mitigate this sparsity, we combine point clouds across five detected frames at a time to obtain a richer representation [56]. The enriching effect of this multiframe combining is demonstrated in Fig. 3 for measurements taken from two occupants. Observing the superposition of points across five frames, the point clouds corresponding to the occupants grow considerably, whereas the point clouds from ambient and multipath reflections remain nearly the same size. This is due to ambient and multipath reflections being less persistent than reflections from an occupant. Thus, proportionally, more points from the desired target are observed as frames are accumulated over time.

The features of each point cloud point used as inputs for our designed DNN are $x \mathrm { \cdot , \mathrm { y \mathrm { - } } } .$ , and z-coordinates, SNR, velocity, and range. The $x \mathrm { \cdot , \ y \mathrm { - } , }$ , and z-coordinates can be obtained from a coordinate transform using range, azimuth, and elevation as

$$
x = R \sin ( \phi ) \cos ( \theta )\tag{16}
$$

$$
y = R \cos ( \phi ) \cos ( \theta )
$$

$$
z = R \sin ( \theta ) .\tag{17}
$$

(18)

The input dimensions are defined as $N _ { T } ~ \times ~ N _ { F } ~ \times ~ N _ { P } ~ \times$ $N _ { S }$ for the number of multiframes (i.e., the number of fiveframe concatenations), number of features, number of points, and number of seats. The number of multiframes is variable depending on the number of detected frames across time. The number of features is fixed at six for each point cloud point. The number of points is variable per multiframe, and padding is used if there are fewer detected point cloud points than the defined maximum for a multiframe. Finally, the number of vehicle seats is fixed at five for the two front and three back seats.

Our network design is outlined in Fig. 4. The network consists of three blocks: the local feature extractor, the global feature extractor, and the decoder. The local feature extractor operates using normalized x-, y-, and z-coordinates to obtain features for each seat. This normalization is based on the center point of each defined seat zone. After a multilayer perceptron (MLP) to project the features of each point to a higher dimensional space, it employs point grouping to capture the local structures of the occupant [41]. The point grouping follows a uniform grid rather than anchor points from the data [38]. Then, another MLP and attention across the points of each group are used to determine a set of features per uniform grid point [38]. In this article, we define attention as follows. For a feature matrix X of size $N _ { a } \times N _ { b }$ , attention across the a dimension is computed as

<!-- image-->  
Fig. 4. Proposed DNN for in-vehicle occupant sensing tasks. The local feature extractor operates using normalized point locations and localized point grouping. The global feature extractor operates across all seat points simultaneously. The decoder operates on the concatenated outputs from the local and global feature extractions to make decisions per seat or across the full car. Dimension sizes are shown for the output of each block in the network. $\bar { N } _ { a } = N _ { x } \times N _ { y } \times N _ { z }$ represents the number of anchors across x-, y-, and z-dimensions for point grouping where $\dot { N } _ { X } = 3 , N _ { Y } = 9$ , and $N _ { Z } = 3 . \ N _ { p } = 8$ is the number of points per anchor, $N _ { D }$ is the number of features of the decoder M $\mathsf { \Pi } _ { \mathsf { - } } \mathsf { P } ,$ and $\bar { N _ { C } }$ is the number of classes. Both $N _ { D }$ and $N _ { C }$ are adjusted based on the occupant sensing task.

$$
\mathbf { Y } ^ { \mathsf { T } } = \mathbf { W } ^ { \mathsf { T } } \mathbf { X }\tag{19}
$$

where

$$
\mathbf { W } = \operatorname { s o f t m a x } \left( \mathbf { X P } \right)\tag{20}
$$

and P is a $N _ { b } \ \times \ 1$ projection matrix of learned weights [38]. The softmax function is used to obtain probability weights across the a dimension.1 The attention output Y is a weighted sum across the first dimension of X using the weights W. This attention mechanism is used across points due to its invariance to point permutations [38], [57]. Finally, at the output of the attention mechanism, a 3-D CNN is employed to extract spatial features for each seat zone, leveraging the uniform point grouping in space [38].

The global feature extractor obtains features from the original $x \mathrm { , ~ } y \mathrm { - } ,$ , and z-coordinates without localized grouping of points. At the input, an MLP with shared weights is first used to project the input features per point to a higher dimensional representation [40]. These features are passed to an attention mechanism to weight extracted features across points and then through another MLP. Extracted features across time are finally passed through a multilayer LSTM to obtain temporal features beyond multiframe combining. Due to the LSTM structure, the network is scalable beyond the maximum number of frames it has observed in training [50]. The employed design of MLPs and attention follows a similar approach to [40] for processing point cloud data. However, rather than use max pooling, we utilize the attention mechanism as proposed by [38]. This combats the sparsity of the point cloud by retaining weighted information across all points rather than only taking the maximum, which would be more suitable for reducing redundant information found in traditional image or video data.

TABLE I  
TI IWR6843ISK-ODS FMCW RADAR CONFIGURATION
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=2>Value</td><td rowspan=1 colspan=3>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Start Frequency</td><td rowspan=1 colspan=2>60 GHz</td><td rowspan=2 colspan=3>Ramp SlopeRamp End Time</td><td rowspan=7 colspan=1>97 MHz/μs41 μs205 μs1 μs11 μs2.2 MHz64</td></tr><tr><td rowspan=2 colspan=1>Maximum RangeRange Resolution</td><td rowspan=1 colspan=2>3.40 m</td></tr><tr><td rowspan=1 colspan=2>5.3 cm</td><td rowspan=2 colspan=3>Idle TimeTransmit Start Time</td></tr><tr><td rowspan=7 colspan=1>Maximum VelocityVelocity ResolutionSweep BandwidthAzimuth FOVElevation FOVTransmit AntennasReceive Antennas</td><td rowspan=1 colspan=2>1.70 m/s</td></tr><tr><td rowspan=1 colspan=2>1.5 cm/s</td><td rowspan=1 colspan=3>ADC Start Time</td></tr><tr><td rowspan=1 colspan=2>2.8 GHz</td><td rowspan=2 colspan=3>Sample RateSamples per Chirp</td></tr><tr><td rowspan=1 colspan=2>120°</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>120°</td><td rowspan=3 colspan=3>Chirps per FrameFrame PeriodicityActive Frame Time</td><td rowspan=1 colspan=1></td><td rowspan=3 colspan=1>220200 ms162 ms</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>4</td></tr></table>

The decoder employs a final MLP with shared weights across time on concatenated local and global features. Then, depending on the occupant sensing task, the structure of the decoder is adjusted. For per-seat classification, max pooling across time is employed after the final MLP, followed by a fully connected (FC) layer and softmax activation function (AF) to obtain class probabilities per seat. Alternatively, for per-seat detection, the sigmoid function replaces the softmax function. For a full-car decision, an additional FC layer is used after the MLP followed by the same max pooling across time, another FC layer, and a softmax operation to obtain class probabilities. This structure for a full-car decision can be used for tasks such as smart CPD, in which we want to detect if one or more children are in the vehicle without an adult.

## V. DATA COLLECTION

Measurements for evaluation are taken using a TI IWR6843ISK-ODS 60 GHz FMCW radar with its parameters and chirp configuration outlined in Table I [45], [58], [59].

<!-- image-->  
(a)  
(b)

Fig. 5. (a) TI IWR6843ISK-ODS radar antenna pattern. (b) Mapped virtual antenna pattern.  
<!-- image-->  
(a)

<!-- image-->  
(b)  
Fig. 6. (a) Radar mounted to the ceiling of the cabin. (b) Physical measurement setup with a baby doll occupying Seat 3.

We additionally provide the antenna configuration of the device in Fig. 5 for both the physical antennas and the mapped virtual array [52], [53], [60]. All measurements were taken in a vehicle with five seats and the radar mounted to the ceiling of the cabin, as shown in Fig. 6. Wires are secured to make sure that they do not cause ambient reflections. The data collected includes 27 participants, consisting of 15 adults, 11 children, and one Bella Rose baby doll from Ashton Drake with realistic breathing and heartbeat movements [61]. Participants occupied the vehicle in groups such that only participants from one group at a time were allowed to occupy the vehicle. This grouping of participants is to facilitate rigorous model evaluation using a leave-one-out approach on a group basis. Participants were instructed only to remain in a designated seat of the vehicle and were allowed to use their phones or talk to each other during measurements to emulate realistic in-vehicle situations. Between one and five occupants fill the five seats of the vehicle at a time, and samples are captured at a rate of 5 FPS for 28 s per trial. The average number of participants in the vehicle at a time was 2.53, i.e., on average, the car was approximately half full. The dataset consists of 657 total 28-s trials across five seats with data distributions for each occupant class provided in Fig. 7.

## VI. EVALUATION

To evaluate our occupant sensing approach, we look at both the preprocessing to obtain the radar point cloud and the model accuracy. For preprocessing, we compare computational complexity and data storage requirements between point cloud generation and full heatmap generation measured by the number of multiplications and number of data points. To evaluate the performance of our occupant sensing model, we look at multiple tasks including localized seat-by-seat detection, classification as baby, child, or adult, classification with detection, and a smart CPD task in which we detect if one or more children are in the vehicle without an adult. For all performance evaluations, we tested six groups totaling 20 participants (11 adults and nine children) unseen by the model during training and validation, aside from the baby doll.

<!-- image-->  
Fig. 7. Distribution of measurements across seats for different occupant classes: adult, child, and baby.

To train the model for each occupant sensing task, we use nine multiframes per decision and the cross entropy loss function with class weights determined by the inverse of class frequencies. Each occupant sensing task is initialized with randomized weights except localized detection, in which we use transfer learning from classification with detection to reduce overfitting. Adam optimizer is used with $\beta _ { 1 } = 0 . 9$ , $\beta _ { 2 } ~ = ~ 0 . 9 9 9$ , a learning rate of 0.001, and weight decay of 0.001 [62]. The model takes approximately two hours to converge using PyTorch and an RTX 3080M GPU. To improve the generalization capabilities of the model, we use two key data augmentations per batch during training. The first augmentation slightly shifts the coordinates of each point. The second augmentation is to create random combinations of occupant point clouds from different seats in training, which we find to improve performance for smart CPD. Specifically, to create a new combination of occupant point clouds, we combine measured point clouds of different seat zones taken from different 28-second trials into a new trial.

## A. Computational Complexity and Data Storage Requirements

The computational complexity is considered for the steps that process the beat signal of (5) to obtain power and velocity per spatial point across five chirps. While heatmap generation assumes a fully populated spatial 3-D map, point cloud generation uses CFAR thresholding to reduce redundant information and zoom-in Capon beamforming to obtain a higher resolution around areas of interest. Directly using the heatmap of (13) reduces the resolution and implies exhaustive velocity spectrum generation through (15). In Fig. 8, we plot the number of multiplications of the heatmap-based approach and the point cloud generation using average observed CFAR detection rates. Comparing the number of multiplications for full heatmap generation to our point cloud approach, we obtain a 54.1% reduction of computational complexity using our employed range FFT size of 64 with 27 range bins selected that correspond to inside the vehicle, 289 $( 1 7 ~ \times ~ 1 7 )$ scan angles across azimuth and elevation dimensions without zoomin beamforming, and a velocity FFT size of 256. Most of this complexity reduction is attributed to skipping velocity spectrum generation for points that are removed after multipass CFAR thresholding. However, even skipping velocity spectrum generation, the increase in complexity to go from the heatmap of (13) to point cloud data using multipass CFAR thresholding is minimal at only 1.1%. Attempting to exhaustively generate a heatmap with the same 1.18◦ resolution of our point cloud with zoom-in Capon beamforming would require approximately 10 201 scanned angles and a 7.1× increase in computational complexity. Finally, we note that the computational complexity of our proposed network is approximately $3 . 0 \times 1 0 ^ { 7 }$ multiplications at maximum for per-seat classification. This is only 21.3% of the complexity for point cloud generation with velocity information. Thus, we determine that the preprocessing approach drives the computational complexity.

<!-- image-->  
Fig. 8. Computational complexity comparisons between heatmap generation and the proposed point cloud generation using average observed detection rates. The cases without velocity estimation per point are also considered.

Considering data storage requirements, we look at the fully populated heatmap of size $2 7 \times 1 7 \times 1 7$ for range, azimuth, and elevation, respectively, and the maximum number of detected points from our dataset at 693 across five frames. The number of features for each heatmap point is two for power and velocity, while the number of features for each point cloud point is six for $x \mathrm { \cdot , \ y \mathrm { - } , }$ , and z-coordinates, SNR, range, and velocity. Comparing the number of points required across five frames, we observe a 94.7% reduction in data storage requirements for the point cloud compared to the heatmap.

## B. Accuracy

In this section, we evaluate the accuracy of our proposed occupant sensing approach across multiple tasks and on groups of participants unseen during model training and validation. These tasks include localized occupant detection, occupant classification, and smart CPD. This additional smart CPD task evaluates the accuracy of our model in making a decision across the full car. We also evaluate the accuracy as it varies with the number of frames, and we compare the performance with and without velocity information to evaluate its significance in occupant sensing tasks.

<!-- image-->  
(a)

<!-- image-->  
(b)

Fig. 9. Localized detection confusion matrices using model-based point cloud thresholding. Right-side tables indicate the corresponding number of accurate and inaccurate detections per row. (a) Confusion matrix for a 45-frame decision. (b) Confusion matrix for a 140-frame decision.  
<!-- image-->  
(a)

<!-- image-->  
(b)  
Fig. 10. Localized detection confusion matrices using our proposed deep learning approach. (a) Confusion matrix for a 45-frame decision. (b) Confusion matrix for a 140-frame decision.

Toward a comprehensive evaluation, we include a comparison to the model-based localized occupant detection approach using defined features and thresholds of the point cloud [36], [37]. For this approach, an occupant is detected based on the number of detected points and the average SNR of the points across a defined number of frames. Exact algorithm steps are outlined in Appendix II. We plot the confusion matrix results in Fig. 9(a) for a 45-frame decision and Fig. 9(b) for a 140-frame decision. Based on the results, we observe an average accuracy of 91.9% for 45 frames and up to 92.7% for 140 frames. Next, using our proposed deep learning approach, we evaluate the performance for the same localized detection task. Confusion matrices are plotted in Fig. 10. From these results, we observe average accuracies of 95.5% for 45 frames and 95.6% for 140 frames. These accuracies achieve improvements of 3.6% and 2.9% for 45-frame and 140-frame decisions, respectively, compared to the model-based point cloud thresholding.

Extending our model to include classification, we evaluate the ability of our approach to distinguish each seat as one of four classes: empty, adult, child, or baby, i.e., simultaneous localized detection and classification. Confusion matrix results for this task are plotted in Fig. 11. From these results, we observe reduced average accuracies of 74.1% for 45 frames and 84.1% for 140 frames, due to the added task of classification. We also notice significant correlations between adult and child classes and empty and baby classes. We next constrain the classification to known occupants as either adult, child, or baby. By focusing on classifying known occupants, we seek to improve the discriminative power of the model. Confusion matrix results for this three-class classification task are plotted in Fig. 12. From these results, we observe improved average accuracies of 88.2% (+14.1%) for 45 frames and 92.4% (+8.3%) for 140 frames. Much of this accuracy improvement is driven by the correct classification of the baby class, especially for the 45-frame decision. Evaluating the performance for the new unseen participant classes, i.e., adult and child, we achieve average accuracies of 82.3% and 88.7% for 45-frame and 140-frame decisions, respectively.

<!-- image-->  
(a)

<!-- image-->  
(b)

Fig. 11. Simultaneous localized detection and classification confusion matrices using our proposed deep learning approach. (a) Confusion matrix for a 45-frame decision. (b) Confusion matrix for a 140-frame decision.  
<!-- image-->  
(a)

<!-- image-->  
(b)

Fig. 12. Occupant classification confusion matrices using our proposed deep learning approach. (a) Confusion matrix for a 45-frame decision. (b) Confusion matrix for a 140-frame decision.  
<!-- image-->  
(a)

<!-- image-->  
(b)  
Fig. 13. Smart CPD confusion matrices using our proposed deep learning approach. The adult present class indicates one or more adults in the vehicle, and the child alone class indicates one or more children in the vehicle without an adult also present. (a) Confusion matrix for a 45-frame decision. (b) Confusion matrix for a 140-frame decision.

Next, we evaluate the performance of our deep learning model for a smart CPD task in which we detect if one or more children are in the vehicle without an adult. This task seeks to evaluate all occupied seats to make a full-car decision. Confusion matrix results for this occupant sensing task are plotted in Fig. 13. From the results, we observe average accuracies of 87.3% for 45 frames and 88.4% for 140 frames.

In real-world implementations, frames can be observed over time to continuously improve the decision. Thus, we look to evaluate the ability of our model to improve its accuracy as more frames become available over time. Moreover, we seek to observe the performance on observation lengths beyond the 45-frame decision the model is trained on. Average accuracy results across different observation lengths are plotted in Fig. 14 in terms of the number of frames. From the results, we observe improvements in accuracy beyond the 45-frame decision that the models are trained on. However, performances appear to plateau beyond roughly 80 frames for all occupant sensing tasks evaluated.

<!-- image-->  
Fig. 14. Accuracy as it varies with the number of observed frames for different occupant sensing tasks.

<!-- image-->  
Fig. 15. Accuracy with and without the inclusion of velocity as an input feature.

Finally, we look to evaluate the performance of our proposed occupant sensing approach with and without velocity as an input. Specifically, for each occupant sensing task, the model is retrained without velocity information, reducing the input feature dimension size by one. This evaluation aims to quantify the significance of including velocity information as an input feature of the network and the benefits of taking a point-cloud-based approach to efficiently include velocity information as discussed in Section VI-A. Average accuracy results for each of the four occupant sensing tasks evaluated are plotted in Fig. 15. From these results, we observe no change in accuracy for the localized detection and 4-class detection with classification. However, we observe notable benefits for the tasks with known occupant locations. For the 3-class classification of each occupant as adult, child, or baby, we observe an improvement of 3% when including velocity information. Likewise, for the smart CPD task, we observe an improvement of 4% when including velocity information.

## VII. CONCLUSION

Occupant sensing using mmWave FMCW radar is challenging due to limited angular resolution, multipath reflections, and ambient reflections in the vehicle. Moreover, identifying and relating reliable features for model-based classification of occupants is difficult when observing radar heatmap or point cloud measurements. To overcome these challenges, we propose a deep learning solution that learns features from the data and works with point cloud information obtained through our multipass CFAR thresholding and zoom-in beamforming. Multipass CFAR thresholding reduces redundancies of fully populated range-angle heatmaps, and zoom-in beamforming improves resolution in detected high-SNR areas. These advantages are obtained while maintaining velocity information, achieving a 54.1% computational complexity reduction and a 94.7% data storage requirements reduction compared to the heatmap-based approach. Furthermore, we assess the benefit of velocity information for occupant classification through accuracy comparisons to the case with velocity information omitted. Using our deep learning model, we evaluate performance across localized occupant detection, occupant classification as adult, child, or baby, and a smart CPD task in which the model determines if one or more children are in the car without an adult. We achieve a localized detection accuracy of 95.6% compared to 92.7% for model-based point cloud thresholding. For classification as adult, child, or baby, we obtain an accuracy of 88.7% on participants unseen by the model during training and validation. Finally, for smart CPD, we achieved an accuracy of 88.4%. These occupant sensing tasks are further evaluated with different time observation sizes, demonstrating robustness. Toward future work, we look to streamline data collection to evaluate data-driven methods across more participants and seating scenarios. We also look toward different radar mounting situations and measurements while the vehicle is moving.

## APPENDIX I MULTIPASS CFAR

In this section, we detail our multipass CFAR algorithm used to detect significant points in space. First, a two-pass CFAR method across the range and azimuth dimensions are outlined in Algorithm 1. Then, zoom-in Capon beamforming with a final CFAR pass is outlined in Algorithm 2. We note that the algorithms in this section are described with indexing across multiple dimensions to improve clarity. However, redundant dimensions and indexing can be omitted in practice for speed. We also define the following which are used in the algorithms.

1) K corresponds to the number of ranges of interest inside the vehicle, rather than the full FFT size. For the results in Section VI, we use an FFT size of 64 and $K = 2 7 .$

2) I and M are the numbers of scanned azimuth and elevation angles, respectively, of the coarse heatmap before zoom-in beamforming. For the results in Section VI, we use $I = 1 7$ and $M = 1 7$

Algorithm 1 Two-Pass CFAR   
Data: Range-angle heatmap: $H _ { k , i , m } ,$ Noise reference   
window sizes: $N _ { W } ^ { ( R ) } , \ N _ { W } ^ { ( \phi ) }$ Guard window sizes:   
$N _ { G } ^ { ( R ) } , N _ { G } ^ { ( \phi ) }$ Range pad averaging size: $N _ { a v e } ^ { ( R ) }$   
Threshold scale factors: $\gamma ^ { ( R ) } , \gamma ^ { \overline { { { ( \phi ) } } } }$   
Result: Detected coarse heatmap indices: $C _ { k , i , m }$   
1 // Pass 1   
2 $N _ { W G }  N _ { W } ^ { ( R ) } + N _ { G } ^ { ( R ) } ;$   
3 for $i = 0 , \ldots , I - \bar { 1 }$ do   
4 for $m = 0 , \ldots , M - 1$ do   
5 $\begin{array} { r } { \mu _ { L } \gets \frac { 1 } { N _ { n v e } ^ { ( R ) } } \sum _ { k = 0 } ^ { N _ { a v e } ^ { ( R ) } - 1 } H _ { k , i , m } ; } \end{array}$   
6 $\begin{array} { r } { \mu _ { R }  \frac { 1 } { N _ { a v e } ^ { ( R ) } } \sum _ { k = K - N _ { a v e } ^ { ( R ) } } ^ { K - 1 } H _ { k , i , m } ; } \end{array}$   
7 $\mathbf { H } \gets \bigl [ H _ { 0 , i , m } ^ { - } , \ldots , H _ { K - 1 , i , m } ^ { - } \bigr ] ^ { \intercal } ;$   
8 $\mathbf { H }  [ \mu _ { L } \mathbf { 1 } _ { N _ { W G } } ^ { \top } , \mathbf { H ^ { \top } } , \mu _ { R } \mathbf { 1 } _ { N _ { W G } } ^ { \top } ] ^ { \top }$   
9 $\left[ { \cal H } _ { 0 } , \ldots , { \cal H } _ { K ^ { \prime } - 1 } \right] ^ { \intercal } = { \bf H } ;$   
10 for $k = 0 , \ldots , \bar { K } ^ { \prime } - 1$ do   
11 $\begin{array} { r } { \mu _ { L }  \frac { 1 } { N _ { W } ^ { ( R ) } } \sum _ { g = k } ^ { k + N _ { W } ^ { ( R ) } - 1 } H _ { g } ; } \end{array}$   
(R) (R)   
12 $\begin{array} { r } { \mu _ { R }  \frac { 1 } { N _ { W } ^ { ( R ) } } \sum _ { g = k + 2 N _ { G } ^ { ( R ) } + N _ { W } ^ { ( R ) } + 1 } ^ { k + 2 N _ { G } ^ { \cdots \prime } + 2 N _ { W } ^ { \cdots \prime } } H _ { g } ; } \end{array}$   
13 $W _ { k , i , m } ^ { ( R ) } \gets \operatorname* { m i n } { ( \mu _ { L } , \mu _ { R } ) } ;$   
14 if $H _ { k , i , m } > \gamma ^ { ( R ) } W _ { k , i , m } ^ { ( R ) }$ then   
15 $| \quad C _ { k , i , m } \gets 1 ;$   
16 else   
17 $C _ { k , i , m } \gets 0 ;$   
18 end   
19 end   
20 end   
21 end   
22 // Pass 2   
23 $N _ { W G }  N _ { W } ^ { ( \phi ) } + N _ { G } ^ { ( \phi ) } ;$   
24 for $k = 0 , \stackrel { \cdots } { \ldots } , K - 1$ do   
25 if $\begin{array} { r } { \sum _ { i = 0 } ^ { I - 1 } \sum _ { m = 0 } ^ { M - 1 } C _ { k , i , m } > 0 } \end{array}$ then   
26 for $m = 0 , \ldots , M - 1$ do   
27 $\mathbf { H } \gets [ H _ { k , I - N _ { W G } , M - 1 } , \dots , H _ { k , I - 1 , M - 1 } ,$   
$H _ { k , 0 , m } , \ldots , H _ { k , I - 1 , m } ,$   
$H _ { k , 0 , 0 } , \ldots , H _ { k , N _ { W G } - 1 , 0 } ] ^ { \mathsf { T } } ;$   
28 $\left[ { { H _ { 0 } } , \ldots , { H _ { I ^ { \prime } - 1 } } } \right] ^ { \intercal } = { \bf { H } } ;$   
29 for $i = 0 , \ldots , \bar { I } ^ { \prime } - 1$ do   
30 $\begin{array} { r } { \mu _ { L }  \frac { 1 } { N _ { W } ^ { ( \phi ) } } \sum _ { g = k } ^ { k + N _ { W } ^ { ( \phi ) } - 1 } H _ { g } ; } \end{array}$   
(φ) (φ)   
31 $\begin{array} { r } { \mu _ { R }  \frac { 1 } { N _ { W } ^ { ( \phi ) } } \sum _ { g = k + 2 N _ { G } ^ { ( \phi ) } + N _ { W } ^ { ( \phi ) } + 1 } ^ { k + 2 N _ { G } ^ { \mathrm { v } \prime } + 2 N _ { W } ^ { \mathrm { v } \prime } } H _ { g } ; } \end{array}$   
32 $W _ { k , i , m } ^ { ( \phi ) } \gets \operatorname* { m i n } { ( \mu _ { L } , \mu _ { R } ) } ;$   
33 if ${ H } _ { k , i , m } > \gamma ^ { ( \phi ) } { W } _ { k , i , m } ^ { ( \phi ) }$ then   
34 $C _ { k , i , m } \gets 1 ;$   
35 else   
36 $C _ { k , i , m } \gets 0 ;$   
37 end   
38 end   
39 end   
40 end   
41 end

Algorithm 2 Zoom-in Capon Beamforming and CFAR   
Pass   
Data: Virtual antenna array covariance: $\mathbf { R } _ { k } ,$ Detected   
coarse heatmap indices: $C _ { k , i , m } ,$ Coarse heatmap   
power: $H _ { k , i , m } ,$ , Coarse heatmap steering vectors:   
$\mathbf { a } _ { i , m } ,$ Zoom-in steering vectors: $\mathbf { b } _ { u , v }$ , Threshold   
shift factor: γ   
Result: Final steering vectors: $\mathbf { c } _ { i , m , u , v } ,$ Heatmap with   
zoom-in angles: $G _ { k , i , m , u , v } ,$ Detected indices   
for the heatmap with zoom-in angles: $J _ { k , i , m , u , v }$   
1 for k = 0, . . . , K − 1 do   
2 for $i = 0 , \ldots , I - 1$ do   
3 for $m = 0 , \ldots , M - 1$ do   
4 if $C _ { k , i , m } = 1$ then   
5 for $u = 0 , \ldots , U - 1$ do   
6 for $v = 0 , \ldots , V ^ { \prime } - 1$ do   
7 $\mathbf { c } _ { i , m , u , v } \gets \mathbf { a } _ { i , m } \odot \mathbf { b } _ { u , v } ;$   
8 $\begin{array} { r } { G _ { k , i , m , u , v }  \frac { 1 } { \mathbf { c } _ { i , m , u , v } ^ { H } \mathbf { R } _ { k } ^ { - 1 } \mathbf { c } _ { i , m , u , v } } ; } \end{array}$   
9 $G _ { m a x } \gets \operatorname* { m a x } _ { u . v } \left( G _ { k , i , m , u , v } \right) ;$   
10 $G _ { m i n } \gets \operatorname* { m i n } _ { u , v } \left( G _ { k , i , m , u , v } \right) ;$   
11 $\begin{array} { r } { \gamma _ { t h }  H _ { k , i , m } ( \gamma - \frac { G _ { m a x } - G _ { m i n } } { G _ { m a x } + G _ { m i n } } ) . } \end{array}$   
12 if $G _ { k , i , m , u , v } > \gamma _ { t h }$ then   
13 $J _ { k , i , m , u , v } \gets 1 ;$   
14 else   
15 $J _ { k , i , m , u , v } \gets 0 ;$   
16 end   
17 end   
18 end   
19 end   
20 end   
21 end   
22 end

3) U and V ′ are the numbers of scanned azimuth and elevation angles, respectively, for zoom-in beamforming. 4) $\mathbf { 1 } _ { N }$ is the $N \times 1$ vector of all ones.

5) min(X, Y ) outputs the minimum value between its inputs X and Y .

6) mi $\mathsf { n } _ { i , k } ( X _ { j , i , k } )$ outputs the minimum across all valid i and k values for the input $X _ { j , i , k }$

7) max ${ } _ { i , k } ( X _ { j , i , k } )$ outputs the maximum across all valid i and k values for the input $X _ { j , i , k }$

The two-pass CFAR algorithm first detects significant points across range bins for each azimuth angle index i and elevation angle index m from Capon beamforming in (13). In lines 5–9, the heatmap is padded in the range dimension to enable estimates of noise for all ranges in $H _ { k , i , m } .$ Then, CFAR thresholding is carried out in lines 10–19. For the current heatmap points of interest $H _ { k , i , m } ,$ surrounding points in the range are used to compute noise level estimates as shown in Fig. 2 to obtain $\mu _ { L } , \ \mu _ { R } ,$ , and $W _ { k , i , m } ^ { ( R ) }$ . Using the noise-level estimate $W _ { k , i , m } ^ { ( R ) }$ and a defined threshold scale factor $\gamma ^ { ( R ) }$ , the detection decision is made in line 14. Next a similar approach is taken for the CFAR pass across azimuth angles with the only change being in how the padding is computed. In line 27, the padding for the azimuth pass uses measurements from the first and last elevation angles scanned rather than measurements from the current elevation index. For thresholding, a new noise estimate $W _ { k , i , m } ^ { ( \phi ) }$ is obtained per azimuth index i in lines 30–32. Finally, the detection decision is made in line 33.

To improve the resolution of the point cloud points, we implement zoom-in capon beamforming with an additional CFAR pass as described in Algorithm 2. For each detection, i.e., $C _ { k , i , m } = 1$ , zoom-in steering vectors

$$
\begin{array} { c } { { { \displaystyle { \bf b } } _ { u , v } = \left[ \exp \left( - j 2 \pi \displaystyle \frac { 1 } { \lambda } ( x _ { 0 } \sin ( \phi _ { u } ) + y _ { 0 } \sin ( \theta _ { v } ) ) \right) , \right. } } \\ { { \left. \exp \left( - j 2 \pi \displaystyle \frac { 1 } { \lambda } ( x _ { 1 } \sin ( \phi _ { u } ) + y _ { 1 } \sin ( \theta _ { v } ) ) \right) , \dots , \right. } } \\ { { \left. \exp \left( - j 2 \pi \displaystyle \frac { 1 } { \lambda } ( x _ { P - 1 } \sin ( \phi _ { u } ) + y _ { P - 1 } \sin ( \theta _ { v } ) ) \right) \right] ^ { \intercal } } } \end{array}\tag{21}
$$

are used to obtain heatmap points $G _ { k , i , m , u , v }$ near the detected point $H _ { k , i , m }$ for $C _ { k , i , m } ~ = ~ 1$ . Then, the maximum $G _ { \mathrm { m a x } }$ and minimum $G _ { \mathrm { m i n } }$ across the zoom-in points are used to form a threshold for CFAR $\begin{array} { r c l } { \gamma _ { \mathrm { t h } } } & { = } & { H _ { k , i , m } ( \gamma \ - } \end{array}$ $( ( G _ { \operatorname* { m a x } } - G _ { \operatorname* { m i n } } ) / ( G _ { \operatorname* { m a x } } + G _ { \operatorname* { m i n } } ) ) )$ where γ is a defined threshold shift factor. Finally, the zoom-in CFAR detection decision is made in line 12.

## APPENDIX IIMODEL-BASED POINT CLOUD THRESHOLDING

Our model-based approach for localized occupant detection is outlined in this section. This point cloud thresholding approach is based on the SNR and number of points detected in each defined seat zone of the vehicle. The exact algorithm steps are described in Algorithm 3. To simplify the notation, we assume a 3-D input for the SNR $S _ { p , f , s }$ across point cloud points. The subscript $p$ denotes the point cloud point index, f denotes the frame index, and s denotes the seat zone index.

The point cloud thresholding approach is based on a state machine $( T _ { f , s } = 1$ for occupied, $T _ { f , s } = 0$ for empty) with a final threshold on line 35 that makes a binary decision across a given number of frames $N _ { f }$ . The state machine operates across frames f and seats s with tunable thresholds per seat. On lines 11–18, the empty state is handled. In this case, we count sub-detections based on the number of detected points $N _ { f , s } ^ { ( P ) }$ and average SNR $S _ { a v g }$ . If the number of sub-detections is met, i.e., $\bar { N } _ { s } ^ { ( T ) } > \gamma _ { s } ^ { ( T ) }$ , then the state is moved to occupied $T _ { f , s } = 1$ , and the number of sub-detections is reset as $N _ { s } ^ { ( { \hat { T } } ) } =$ 0. On lines 19–30, the occupied state is handled. The same thresholds based on the number of detected points and average SNR are used to reset ${ N _ { s } ^ { ( T ) } }$ to zero. However, if the number of detected points $N _ { f , s } ^ { ( P ) }$ falls below the minimum threshold $\gamma _ { s } ^ { ( M ) }$ , then ${ N _ { s } ^ { ( T ) } }$ is incremented to count nondetections. Once $\dot { N } _ { s } ^ { ( T ) }$ exceeds the forget threshold $\gamma _ { s } ^ { ( F ) } ,$ , then the state is reset to the empty $T _ { f , s } = 0$ state, and $\dot { N } _ { s } ^ { ( T ) }$ is reset to 0. To use the states to make a binary decision across a given number of frames $N _ { f } .$ , we count the number of occupied states across frames on line 34 and use the final threshold on line 35.

Algorithm 3 Localized Occupant Detection by Point   
Cloud Thresholding   
Data: Number of frames for decision: $N _ { f }$ Number of   
seat zones: $N _ { S }$ Number of detected points: ${ \cal N } _ { f , s } ^ { ( P ) }$   
SNR of detected points: $S _ { p , f , s }$ Number of   
points detected threshold: $\gamma _ { s } ^ { ( { P } ) }$ Average SNR   
threshold: $\gamma _ { s } ^ { ( S ) }$ Sub-detections threshold: $\gamma _ { s } ^ { ( T ) }$   
Forget threshold: $\gamma _ { s } ^ { ( F ) }$ Minimum number of   
points threshold: $\dot { \gamma } _ { s } ^ { ( M ) }$ Detection rate threshold:   
$\overset { \cdot } { \gamma } { } ^ { ( R T ) }$   
Result: Binary occupancy decisions across seats: $O _ { s }$   
1 $\begin{array} { r } { N _ { s } ^ { ( T ) } \gets 0 , \ s = 0 , \ldots , N _ { S } ; } \end{array}$   
2 $T _ { - 1 , s } \gets 0 , s = 0 , \ldots , N _ { S } ;$   
3 for $f = 0 , \dots , N _ { f } - 1$ do   
4 for $s = 0 , \ldots , N _ { S }$ do   
5 $T _ { f , s } \gets T _ { f - 1 , s } ;$   
6 if $N _ { f , s } ^ { ( P ) } > 0$ then   
7 $\begin{array} { r } {  S _ { a v g }  \frac { 1 } { N _ { f , s } ^ { ( P ) } } \sum _ { p = 0 } ^ { N _ { f , s } ^ { ( P ) } - 1 } S _ { p , f , s } ;  } \end{array}$   
8 else   
9 $S _ { a v g }  - 1 ;$   
10 end   
11 if $T _ { f , s } = 0$ then   
12 if $N _ { f , s } ^ { ( P ) } > \gamma _ { s } ^ { ( P ) }$ and $S _ { a v g } > \gamma _ { s } ^ { ( S ) }$ then   
13 $\begin{array} { r } { \dot { N } _ { s } ^ { ( T ) }  N _ { s } ^ { ( T ) } + 1 ; } \end{array}$   
14 if $N _ { s } ^ { ( T ) } > \gamma _ { s } ^ { ( T ) }$ then   
15 $T _ { f , s } \gets 1 ;$   
16 $N _ { s } ^ { ( T ) } \gets 0 ;$   
17 end   
18 end   
19 else if $T _ { f , s } = 1$ then   
20 if $N _ { f , s } ^ { ( { P } ) } > \gamma _ { s } ^ { ( { P } ) }$ )and $\boldsymbol { S _ { a v g } } > \gamma _ { s } ^ { ( S ) }$ then   
21 $\begin{array} { r } { N _ { s } ^ { ( T ) }  0 ; } \end{array}$   
22 else if $N _ { f , s } ^ { ( P ) } \leq \gamma _ { s } ^ { ( M ) }$ then   
23 if $N _ { s } ^ { ( { \dot { T } } ) } > \gamma _ { s } ^ { ( F ) }$ then   
24 $T _ { f , s } \gets 0 ;$   
25 $N _ { s } ^ { ( T ) } \gets 0 ;$   
26 else   
27 $N _ { s } ^ { ( T ) } \gets N _ { s } ^ { ( T ) } + 1 ;$   
28 end   
29 end   
30 end   
31 end   
32 end   
33 for $s = 0 , \ldots , N _ { S }$ do   
34 $\begin{array} { r } { \eta = \sum _ { f = 0 } ^ { N _ { f } - 1 } T _ { f , s } ; } \end{array}$   
35 if $\eta > \dot { \gamma } ^ { ( R T ) } N _ { f }$ then   
36 $O _ { s } \gets 1 ; / /$ Occupied   
37 else   
38 $O _ { s } \gets 0 ; / /$ Empty   
39 end   
40 end

## REFERENCES

[1] A. R. Diewald et al., “RF-based child occupation detection in the vehicle interior,” in Proc. Int. Radar Symp., 2016, pp. 1–4.

[2] (Jan. 2024). Heatstroke Deaths of Children in Vehicles. Accessed: Apr. 30, 2024. [Online]. Available: https://www.noheatstroke.org

[3] L. Glenn, E. Glenn, and L. Neurauter, “Pediatric vehicle heatstroke: Evaluation of preventative technologies,” Virginia Tech Transp. Inst., Blacksburg, VA, USA, Tech. Rep. 21-UT-098, Apr. 2021.

[4] (Sep. 2023). Euro NCAP 2025 Roadmap. Accessed: Apr. 30, 2024. [Online]. Available: https://cdn.euroncap.com/media/30700/euroncaproadmap-2025-v4.pdf

[5] (Aug. 2021). NHTSA Proposes Seat Belt Warning Expansion To Encourage Drivers and Passengers To Buckle Up. Accessed: Apr. 30, 2024. [Online]. Available: https://www.nhtsa.gov/pressreleases/nhtsa-proposes-seat-belt-warning-system-expansion

[6] X. Zeng, F. Wang, B. Wang, C. Wu, K. J. R. Liu, and O. C. Au, “In-vehicle sensing for smart cars,” IEEE Open J. Veh. Technol., vol. 3, pp. 221–242, 2022.

[7] M. K. Arsath Ali and R. Ramli, “Prevention alert system for a child left in a parked vehicle,” J. Manage. Sci., vol. 19, no. 2, p. 13, Dec. 2021.

[8] K. Kasten, A. Stratmann, M. Munz, K. Dirscherl, and S. Lamers, “iBolt technology—A weight sensing system for advanced passenger safety,” in Advanced Microsystems for Automotive Applications 2006. Pfinztal, Germany: Fraunhofer Instituet Fuer Chemishce Technologie (ICT), 2006, pp. 171–186.

[9] A. Voisin, S. Bombardier, E. Levrat, and J. Bremont, “Sensory features measurement of the under-thigh length of car seat,” in Proc. IEEE World Congr. Comput. Intell., vol. 2, May 1998, pp. 1589–1594.

[10] B. George, H. Zangl, T. Bretterklieber, and G. Brasseur, “Seat occupancy detection based on capacitive sensing,” IEEE Trans. Instrum. Meas., vol. 58, no. 5, pp. 1487–1494, May 2009.

[11] A. Ranjan and B. George, “A child-left-behind warning system based on capacitive sensing principle,” in Proc. IEEE Int. Instrum. Meas. Technol. Conf. (I2MTC), May 2013, pp. 702–706.

[12] D. Tumpold and A. Satz, “Contactless seat occupation detection system based on electric field sensing,” in Proc. 35th Annu. Conf. IEEE Ind. Electron., Nov. 2009, pp. 1823–1828.

[13] M. Walter, B. Eilebrecht, T. Wartzek, and S. Leonhardt, “The smart car seat: Personalized monitoring of vital signs in automotive applications,” Pers. Ubiquitous Comput., vol. 15, no. 7, pp. 707–715, Oct. 2011.

[14] P. Zappi, E. Farella, and L. Benini, “Tracking motion direction and distance with pyroelectric IR sensors,” IEEE Sensors J., vol. 10, no. 9, pp. 1486–1494, Sep. 2010.

[15] F. R. M. Rashidi and I. H. Muhamad, “Vehicle’s interior movement detection and notification system,” Recent Adv. Autom. Control Model. Simul., vol. 2013, pp. 139–144, Apr. 2013.

[16] M. Devy, A. Giralt, and A. Marin-Hernandez, “Detection and classification of passenger seat occupancy using stereovision,” in Proc. IEEE Intell. Vehicles Symp., Oct. 2000, pp. 714–719.

[17] P. Kuchár, R. Pirník, T. Tichý, K. Rástocný, M. Skuba, and T. Tettamanti,ˇ “Noninvasive passenger detection comparison using thermal imager and IP cameras,” Sustainability, vol. 13, no. 22, p. 12928, Nov. 2021.

[18] I. Papakis, A. Sarkar, A. Svetovidov, J. S. Hickman, and A. L. Abbott, “Convolutional neural network-based in-vehicle occupant detection and classification method using second strategic highway research program cabin images,” Transp. Res. Rec., J. Transp. Res. Board, vol. 2675, no. 8, pp. 443–457, Apr. 2021.

[19] M. Fritzsche, C. Prestele, G. Becker, M. Castillo-Franco, and B. Mirbach, “Vehicle occupancy monitoring with optical range-sensors,” in Proc. IEEE Intell. Vehicles Symp., Jun. 2004, pp. 90–94.

[20] A. Gharamohammadi, A. Khajepour, and G. Shaker, “In-vehicle monitoring by radar: A review,” IEEE Sensors J., vol. 23, no. 21, pp. 25650–25672, Nov. 2023.

[21] X. Zeng, B. Wang, C. Wu, S. D. Regani, and K. J. R. Liu, “WiCPD: Wireless child presence detection system for smart cars,” IEEE Internet Things J., vol. 9, no. 24, pp. 24866–24881, Dec. 2022.

[22] Y. Ma, Y. Zeng, and V. Jain, “CarOSense: Car occupancy sensing with the ultra-wideband keyless infrastructure,” Proc. ACM Interact., Mobile, Wearable Ubiquitous Technol., vol. 4, no. 3, pp. 1–28, Sep. 2020.

[23] J. Möderl, F. Pernkopf, and K. Witrisal, “Car occupancy detection using ultra-wideband radar,” in Proc. 18th Eur. Radar Conf. (EuRAD), Apr. 2022, pp. 313–316.

[24] F. Fioranelli, E. Rufas, and A. Yarovoy, “Multiple people detection & localization with multistatic UWB radar in vehicle cabin,” in Proc. IEEE Int. Workshop Antenna Technol. (iWAT), Apr. 2024, pp. 277–280.

[25] S. Lim, S. Lee, J. Jung, and S.-C. Kim, “Detection and localization of people inside vehicle using impulse radio ultra-wideband radar sensor,” IEEE Sensors J., vol. 20, no. 7, pp. 3892–3901, Apr. 2020.

[26] S. Lim, J. Jung, S.-C. Kim, and S. Lee, “Deep neural network-based invehicle people localization using ultra-wideband radar,” IEEE Access, vol. 8, pp. 96606–96612, 2020.

[27] S.-Y. Kwon and S. Lee, “In-vehicle seat occupancy detection using ultra-wideband radar sensors,” in Proc. 23rd Int. Radar Symp. (IRS), Sep. 2022, pp. 275–278.

[28] X. Yang, Y. Ding, X. Zhang, and L. Zhang, “Spatial–temporal-circulated GLCM and physiological features for in-vehicle people sensing based on IR-UWB radar,” IEEE Trans. Instrum. Meas., vol. 71, pp. 1–13, 2022.

[29] J. Möderl, S. Posch, F. Pernkopf, and K. Witrisal, “UWBCarGraz dataset for car occupancy detection using ultra-wideband radar,” in Proc. IEEE Radar Conf. (RadarConf), May 2024, pp. 1–6.

[30] Texas Instruments. (Oct. 2024). mmWave Radar Sensor Design & Development. Accessed: Jul. 23, 2024. [Online]. Available: https:// www.ti.com/design-development/embedded-development/mmWaveradar.html

[31] B. R. Upadhyay, A. B. Baral, and M. Torlak, “Vital sign detection via angular and range measurements with mmWave MIMO radars: Algorithms and trials,” IEEE Access, vol. 10, pp. 106017–106032, 2022.

[32] M. Hoffmann, D. Tatarinov, J. Landwehr, and A. R. Diewald, “A fourchannel radar system for rear seat occupancy detection in the 24 GHz ISM band,” in Proc. German Microw. Conf., Mar. 2018, pp. 95–98.

[33] H. Song, Y. Yoo, and H.-C. Shin, “In-vehicle passenger detection using FMCW radar,” in Proc. Int. Conf. Inf. Netw. (ICOIN), Jan. 2021, pp. 644–647.

[34] H. Song and H.-C. Shin, “Single-channel FMCW-radar-based multipassenger occupancy detection inside vehicle,” Entropy, vol. 23, no. 11, p. 1472, Nov. 2021.

[35] H. Abedi, C. Magnier, V. Mazumdar, and G. Shaker, “Improving passenger safety in cars using novel radar signal processing,” Eng. Rep., vol. 3, no. 12, May 2021, Art. no. e12413.

[36] W. Li, Y. Gao, Z. Hu, N. Liu, K. Wang, and S. Niu, “In-vehicle occupant detection system using mm-wave radar,” in Proc. 7th Int. Conf. Commun., Image Signal Process. (CCISP), Nov. 2022, pp. 395–399.

[37] N. Munte, A. Lazaro, R. Villarino, and D. Girbau, “Vehicle occupancy detector based on FMCW mm-wave radar at 77 GHz,” IEEE Sensors J., vol. 22, no. 24, pp. 24504–24515, Dec. 2022.

[38] H. Xue et al., “mmMesh: Towards 3D real-time dynamic human mesh construction using millimeter-wave,” in Proc. 19th Annu. Int. Conf. Mobile Syst. Appl. Services, 2021, pp. 269–282.

[39] H. Xue et al., “M4esh: mmWave-based 3D human mesh construction for multiple subjects,” in Proc. 20th ACM Conf. Embedded Netw. Sensor Syst., Jan. 2023, pp. 391–406.

[40] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “PointNet: Deep learning on point sets for 3D classification and segmentation,” CoRR, vol. abs/1612.00593, pp. 1–19, Dec. 2016.

[41] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “PointNet++: Deep hierarchical feature learning on point sets in a metric space,” CoRR, vol. abs/1706.02413, pp. 1–14, Jun. 2017.

[42] D. Lu, Q. Xie, M. Wei, K. Gao, L. Xu, and J. Li, “Transformers in 3D point clouds: A survey,” 2022, arXiv:2205.07417.

[43] G. Paterniani et al., “Radar-based monitoring of vital signs: A tutorial overview,” Proc. IEEE, vol. 111, no. 3, pp. 277–317, Mar. 2023.

[44] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Jun. 2016, pp. 770–778.

[45] S. Rao. (Apr. 2017). mmWave Radar Sensors. Texas Instruments. Accessed: Apr. 30, 2024. [Online]. Available: https://www.ti.com/video/series/mmWave-training-series.html

[46] B. R. Upadhyay, “Vital signs detection using mmWave MIMO radars: Algorithms and applications,” Ph.D. dissertation, Univ. Texas Dallas, Richardson, TX, USA, Aug. 2023.

[47] M. Richards, Fundamentals of Radar Signal Processing. New York, NY, USA: McGraw-Hill, Oct. 2005.

[48] H. Abedi, S. Luo, V. Mazumdar, M. M. Y. R. Riad, and G. Shaker, “AI-powered in-vehicle passenger monitoring using low-cost mm-wave radar,” IEEE Access, vol. 10, pp. 18998–19012, 2022.

[49] H. Abedi, M. Ma, J. He, J. Yu, A. Ansariyan, and G. Shaker, “Deep learning-based in-cabin monitoring and vehicle safety system using a 4-D imaging radar sensor,” IEEE Sensors J., vol. 23, no. 11, pp. 11296–11307, Jun. 2023.

[50] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, pp. 1735–1780, Nov. 1997.

[51] J. P. van Marter, A. V. Mani, A. G. Dabak, S. Rao, and M. Torlak, “CNN-based in-vehicle occupant sensing using millimeter-wave radar,” in Proc. IEEE Radar Conf. (RadarConf), May 2024, pp. 1–6.

[52] M. E. Yanik, D. Wang, and M. Torlak, “3-D MIMO-SAR imaging using multi-chip cascaded millimeter-wave sensors,” in Proc. IEEE Global Conf. Signal Inf. Process. (GlobalSIP), Nov. 2019, pp. 1–5.

[53] M. E. Yanik and M. Torlak, “Near-field MIMO-SAR millimeter-wave imaging with sparsely sampled aperture data,” IEEE Access, vol. 7, pp. 31801–31819, 2019.

[54] W. Carrara, R. Goodman, and R. Majewski, Spotlight Synthetic Aperture Radar: Signal Processing Algorithms (Artech House Remote Sensing Library). Norwood, MA, USA: Artech House, 1995.

[55] J. Capon, “High-resolution frequency-wavenumber spectrum analysis,” Proc. IEEE, vol. 57, no. 8, pp. 1408–1418, Aug. 1969.

[56] S. An and U. Y. Ogras, “Fast and scalable human pose estimation using mmWave point cloud,” in Proc. 59th ACM/IEEE Design Autom. Conf., Jul. 2022, pp. 889–894.

[57] M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Poczos, R. R. Salakhutdinov, and A. J. Smola et al., “Deep sets,” in Proc. Adv. Neural Inf. Process. Syst., vol. 30, Dec. 2017, pp. 3394–3404.

[58] Texas Instruments. (Oct. 2023). IWR6843ISK-ODS: IWR6843 Intelligent mmWave Overhead Detection Sensor (ODS) Antenna Plug-in Module. Accessed: Apr. 30, 2024. [Online]. Available: https://www.ti.com/tool/IWR6843ISK-ODS

[59] V. Dham. (Feb. 2020). Programming Chirp Parameters in TI Radar Devices. Accessed: Apr. 30, 2024. [Online]. Available: https://www.ti.com/lit/pdf/swra553

[60] (2022). 60 GHz mmWave Sensor EVMs. Accessed: Apr. 30, 2024. [Online]. Available: https://www.ti.com/lit/pdf/swru546

[61] (2023). Bella Rose Baby Doll. Accessed: Apr. 30, 2024. [Online]. Available: https://www.ashtondrake.com/products/301881001_lifelikebreathing-baby-doll.html

[62] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” 2014, arXiv:1412.6980.

<!-- image-->

Jayson P. Van Marter (Student Member, IEEE) received the B.S. (summa cum laude) and Ph.D. degrees in electrical engineering from The University of Texas at Dallas, Richardson, TX, USA, in 2020 and 2024, respectively.

In 2020, he developed a USRP softwaredefined radio testbed engine as a TxACE Intern. In 2023, he developed algorithms for automotive in-cabin sensing using millimeter-wave radar as an Intern with Texas Instruments, Dallas, TX, USA. He has been with Texas Instruments

Radar Systems and Algorithms research and development since 2024. His current research interests include real-time embedded systems, localization, millimeter-wave radar, terahertz radar, SAR imaging algorithms, and radar-human sensing.

Dr. Van Marter received the TxACE Promising Researcher Award in May 2020, the Jonsson School Excellence in Education Doctoral Fellowship in August 2020, the Jan P. Van der Ziel Fellowship in April 2023, and the TxACE Symposium Best Poster Award in October 2023.

<!-- image-->

Anand G. Dabak (Fellow, IEEE) received the bachelor’s degree from IIT Bombay, Mumbai, India, in 1987, and the master’s and Ph.D. degrees in electrical engineering from Rice University, Houston, TX, USA, in 1989 and 1992, respectively.

He joined the DSP Systems Research and Development Center, TI, Dallas, TX, USA, as a Member of the Technical Staff working on wireless systems, in 1995. He worked until 2011 on algorithms, standards, and systems issues related to communications, namely 3GPP, WCDMA, LTE, UWB, powerline communications (PLC), and modem development on TI processors. From 2011 to 2019, he worked within Kilby Labs on ultrasonic flow metering for residential water and gas metering. From 2019 to 2021, he developed localization solutions for Bluetooth low energy (BLE) systems using the angle of arrival (AoA) and employed super-resolution techniques for high-accuracy distance measurement (HADM) phasebased techniques. Since 2021, he has been working in the radar group on applying signal processing techniques to radar applications. He has more than 250 patents in the areas of signal processing for wireless, PLC, and ultrasound applications to flow metering.

Dr. Dabak has been a TI Fellow since 2007.

<!-- image-->

Anil Varghese Mani (Member, IEEE) received the master’s degree in signal processing from IIT Madras, Chennai, India, in 2008.

Since 2008, he has been working with TI, Dallas, TX, USA, in different algorithm and signal processing roles, where he is currently a Systems and Algorithms Manager with the Radar Business Unit. His research interests include information theory, signal processing (specifically radar signal processing), and embedded processing.

<!-- image-->

Sandeep Rao (Senior Member, IEEE) received the bachelor’s degree from IIT Madras, Chennai, India, and the master’s degree from the University of Maryland, College Park, MD, USA.

Prior to TI, he was with Hughes Network Systems, where he worked on modems for Satellite Communication. He is currently with Texas Instruments, Dallas, TX, USA, leading the mmWave sensing Algorithm Group. He has more than 30 patents in the area of mmWave Radar and GNSS positioning. His research interests include radar signal processing, including automotive radar, interference mitigation strategies, and classification.

<!-- image-->

Murat Torlak (Senior Member, IEEE) received the M.S. and Ph.D. degrees in electrical engineering from The University of Texas at Austin, Austin, TX, USA, in 1995 and 1999, respectively.

Since August 1999, he has been with the Department of Electrical and Computer Engineering, The University of Texas at Dallas, Richardson, TX, USA, where he has been promoted to the rank of a Full Professor. He is serving as a Rotating Program Director with the

U.S. National Science Foundation (NSF). His current research interests include experimental verification of wireless networking systems, cognitive radios, millimeter-wave automotive radars, millimeterwave imaging systems, and interference mitigation in radio telescopes.

Dr. Torlak was the General Chair of Symposium on Millimeter Wave Imaging and Communications, in 2013 IEEE GlobalSIP Conference. He was an Associate Editor of IEEE TRANSACTIONS ON WIRELESS COMMUNICATIONS, from 2008 to 2013. He was a Guest Coeditor for the Special Issue on Recent Advances in Automotive Radar Signal Processing of IEEE JOURNAL OF SELECTED TOPICS IN SIGNAL PROCESSING (JSTSP), in 2021.