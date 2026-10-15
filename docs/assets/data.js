/* All numbers come from the thesis/paper results (paper_results_section.tex,
   supplementary.tex, statistical_analysis_results/extended_statistical_results.json,
   medvit_v2_dual_eval_results). Text is bilingual: {en, el}. */
window.DATA = (function () {

  const CLASSES = [
    { id: 'glioma',     en: 'Glioma',     el: 'Γλοίωμα',     color: '--c-glioma' },
    { id: 'meningioma', en: 'Meningioma', el: 'Μηνιγγίωμα',  color: '--c-meningioma' },
    { id: 'no_tumor',   en: 'No tumour',  el: 'Χωρίς όγκο',  color: '--c-notumor' },
    { id: 'pituitary',  en: 'Pituitary',  el: 'Υπόφυση',     color: '--c-pituitary' },
  ];

  // Table 1 of the paper: 503-sample tumour-positive subset.
  const MODELS = [
    { n: 1, id: 'binary',    name: 'Binary CNN',           acc: 98.60, lo: 97.2, hi: 99.3, f1: 98.60, auc: 0.999,  params: 1.44,  inf: 281.6, infSd: 31.3, rank: 4.757, fam: 'cnn' },
    { n: 2, id: 'cnn4',      name: '4-Class CNN',          acc: 98.01, lo: 96.4, hi: 98.9, f1: 98.00, auc: 0.998,  params: 1.28,  inf: 287.4, infSd: 27.7, rank: 4.945, fam: 'cnn' },
    { n: 3, id: 'cnn3',      name: '3-Class CNN',          acc: 97.80, lo: 96.1, hi: 98.8, f1: 97.80, auc: 0.997,  params: 1.28,  inf: 270.5, infSd: 27.1, rank: 4.856, fam: 'cnn' },
    { n: 4, id: 'ensemble',  name: '1-vs-1 Ensemble',      acc: 97.60, lo: 95.9, hi: 98.6, f1: 97.60, auc: 0.997,  params: 3.84,  inf: null,  infSd: null, rank: 5.053, fam: 'cnn' },
    { n: 5, id: 'resnet',    name: 'ResNet-50 Dual',       acc: 76.85, lo: 73.0, hi: 80.3, f1: 76.80, auc: 0.952,  params: 26.1,  inf: 295.2, infSd: 28.6, rank: 5.339, fam: 'tl' },
    { n: 6, id: 'efnet',     name: 'EfficientNet-B3 Dual', acc: 86.03, lo: 82.7, hi: 88.8, f1: 86.00, auc: 0.974,  params: 12.8,  inf: 294.3, infSd: 30.1, rank: 5.097, fam: 'tl' },
    { n: 7, id: 'multidual', name: 'Multi-Dual System',    acc: 93.41, lo: 90.9, hi: 95.3, f1: 93.40, auc: 0.991,  params: 1.68,  inf: 269.5, infSd: 26.8, rank: 4.990, fam: 'cnn' },
    { n: 8, id: 'medvit',    name: 'MedViT V2 Base',       acc: 93.03, lo: 90.5, hi: 95.0, f1: 92.80, auc: 0.992,  params: 49.31, inf: 289.2, infSd: 26.0, rank: 5.124, fam: 'vit' },
    { n: 9, id: 'medvitdual',name: 'MedViT V2 Dual',       acc: 98.15, lo: 96.6, hi: 99.0, f1: 98.13, auc: 0.9998, params: 49.94, inf: 303.2, infSd: 31.1, rank: 4.838, fam: 'vit' },
  ];

  // Table 2: per-class F1 on the full 703-image test set.
  const F1 = {
    models: ['4-Class CNN', 'MedViT V2 Base', 'MedViT V2 Dual'],
    values: {
      glioma:     [98.75, 92.41, 98.76],
      meningioma: [97.58, 88.02, 97.21],
      no_tumor:   [99.26, 97.00, 98.75],
      pituitary:  [99.15, 93.77, 97.78],
    },
  };

  // Confusion matrix, MedViT V2 Dual, 703 images. Rows: true; cols: predicted.
  const CM = [
    [159, 0, 0, 3],
    [1, 157, 3, 4],
    [0, 1, 198, 1],
    [0, 0, 0, 176],
  ];

  // Table S4: bootstrap occlusion confidence drop.
  const BOOT = [
    { model: '4-Class CNN',       g: [5.8, 3.6, 8.3, 160],   m: [64.1, 58.1, 69.9, 156], p: [1.3, 0.3, 2.8, 176] },
    { model: '3-Class CNN',       g: [11.0, 7.6, 14.7, 161], m: [45.8, 39.5, 52.0, 162], p: [6.0, 4.4, 7.8, 175] },
    { model: 'Multi-Dual System', g: [8.8, 5.2, 13.0, 156],  m: [51.7, 45.0, 58.3, 147], p: [1.8, 0.4, 3.6, 176] },
    { model: 'MedViT V2 Base',    g: [16.7, 12.5, 21.4, 147],m: [51.1, 45.7, 56.6, 157], p: [13.8, 10.6, 17.2, 176] },
    { model: 'MedViT V2 Dual',    g: [2.1, 0.9, 3.7, 162],   m: [66.7, 60.6, 72.7, 161], p: [0.6, 0.1, 1.6, 176] },
  ];

  // Table S3: 5-fold stratified CV.
  const CV = {
    models: ['4-Class CNN', '3-Class CNN', 'Binary CNN', 'Multi-Dual System'],
    folds: [[97.39, 97.12, 99.53, 98.10], [98.18, 97.01, 99.84, 98.10], [98.10, 98.12, 99.68, 96.76], [98.02, 98.01, 99.45, 97.78], [98.26, 98.34, 98.89, 98.18]],
    mean: [97.99, 97.72, 99.48, 97.78],
    sd: [0.31, 0.54, 0.32, 0.53],
  };

  // Pairwise matrices (order = MODELS order). p: Bonferroni-adjusted Wilcoxon; d: Cohen's d.
  const PAIR = {
    p: [[null, 0.00043, 0.162, 2.08e-06, 2.7e-14, 6.75e-08, 1.23e-05, 1.45e-08, 0.24], [0.00043, null, 1.0, 1.0, 1.48e-07, 0.0815, 1.0, 0.0925, 0.0192], [0.162, 1.0, null, 0.00581, 2.09e-09, 0.00205, 0.192, 0.000536, 1.0], [2.08e-06, 1.0, 0.00581, null, 0.00174, 1.0, 1.0, 1.0, 0.000424], [2.7e-14, 1.48e-07, 2.09e-09, 0.00174, null, 0.00563, 8.62e-06, 0.0145, 1.74e-11], [6.75e-08, 0.0815, 0.00205, 1.0, 0.00563, null, 1.0, 1.0, 6.71e-05], [1.23e-05, 1.0, 0.192, 1.0, 8.62e-06, 1.0, null, 0.913, 0.00346], [1.45e-08, 0.0925, 0.000536, 1.0, 0.0145, 1.0, 0.913, null, 3.47e-06], [0.24, 0.0192, 1.0, 0.000424, 1.74e-11, 6.71e-05, 0.00346, 3.47e-06, null]],
    d: [[null, 0.27, 0.181, 0.354, 0.528, 0.384, 0.307, 0.402, 0.159], [0.27, null, 0.107, 0.102, 0.31, 0.14, 0.045, 0.161, 0.132], [0.181, 0.107, null, 0.206, 0.404, 0.242, 0.151, 0.262, 0.026], [0.354, 0.102, 0.206, null, 0.212, 0.038, 0.057, 0.059, 0.23], [0.528, 0.31, 0.404, 0.212, null, 0.175, 0.267, 0.153, 0.425], [0.384, 0.14, 0.242, 0.038, 0.175, null, 0.095, 0.022, 0.265], [0.307, 0.045, 0.151, 0.057, 0.267, 0.095, null, 0.117, 0.176], [0.402, 0.161, 0.262, 0.059, 0.153, 0.022, 0.117, null, 0.285], [0.159, 0.132, 0.026, 0.23, 0.425, 0.265, 0.176, 0.285, null]],
  };

  const STATS = { chi2: 164.45, df: 8, p: '1.9 × 10⁻³¹', cd: 0.536, alpha: 0.05, pairs: 36, bonf: 0.00139 };

  /* ───────── Pipeline flow ───────── */
  const PIPELINE = {
    nodes: [
      { id: 'data',  x: 0, y: 0, kind: 'train', en: ['Dataset', '7,023 MRI · 4 classes'], el: ['Δεδομένα', '7.023 MRI · 4 κλάσεις'],
        d: { en: ['Brain Tumor MRI Dataset (Kaggle)', 'A public collection of 2D brain MRI slices in axial, coronal and sagittal planes, labelled as glioma, meningioma, pituitary adenoma or no tumour. All nine models are trained and tested on the same images.'],
             el: ['Brain Tumor MRI Dataset (Kaggle)', 'Δημόσια συλλογή δισδιάστατων τομών MRI εγκεφάλου σε αξονικό, στεφανιαίο και οβελιαίο επίπεδο, με ετικέτες γλοίωμα, μηνιγγίωμα, αδένωμα υπόφυσης ή χωρίς όγκο. Και τα εννέα μοντέλα εκπαιδεύονται και ελέγχονται στις ίδιες εικόνες.'],
             specs: ['7,023 images', 'glioma · meningioma · pituitary · no tumour'] } },
      { id: 'prep',  x: 1, y: 0, kind: 'train', en: ['Preprocessing', 'resize · normalise · augment'], el: ['Προεπεξεργασία', 'αλλαγή μεγέθους · κανονικοποίηση · επαύξηση'],
        d: { en: ['Preprocessing & augmentation', 'Every slice is resized to 224 × 224 × 3 and scaled to [0, 1]. Training images are augmented on the fly with conservative transforms that keep anatomy plausible: brain symmetry allows horizontal flips, but rotations stay small.'],
             el: ['Προεπεξεργασία & επαύξηση', 'Κάθε τομή αλλάζει μέγεθος σε 224 × 224 × 3 και κανονικοποιείται στο [0, 1]. Οι εικόνες εκπαίδευσης επαυξάνονται δυναμικά με συντηρητικούς μετασχηματισμούς που κρατούν την ανατομία ρεαλιστική: η συμμετρία του εγκεφάλου επιτρέπει οριζόντια αναστροφή, οι περιστροφές όμως μένουν μικρές.'],
             specs: ['224×224×3', 'rotation ±15°', 'zoom 0.1', 'brightness 0.9–1.1', 'horizontal flip'] } },
      { id: 'split', x: 2, y: 0, kind: 'train', en: ['Stratified split', '70 / 20 / 10 %'], el: ['Στρωματοποιημένος διαχωρισμός', '70 / 20 / 10 %'],
        d: { en: ['Train / validation / test split', 'A stratified split keeps the class proportions identical in every subset. The 10% test set (703 images) is frozen: it is never seen during training, model selection or cross-validation.'],
             el: ['Διαχωρισμός εκπαίδευσης / επικύρωσης / ελέγχου', 'Ο στρωματοποιημένος διαχωρισμός κρατά τις αναλογίες των κλάσεων ίδιες σε κάθε υποσύνολο. Το 10% ελέγχου (703 εικόνες) είναι «παγωμένο»: δεν χρησιμοποιείται ποτέ στην εκπαίδευση, στην επιλογή μοντέλου ή στο cross-validation.'],
             specs: ['train 70%', 'validation 20%', 'test 10% = 703', 'stratified'] } },
      { id: 'train', x: 3, y: 0, kind: 'train', en: ['9 architectures', 'CNN · ensemble · TL · ViT'], el: ['9 αρχιτεκτονικές', 'CNN · ensemble · TL · ViT'],
        d: { en: ['Training nine architectures', 'Custom CNNs, a one-vs-one ensemble, two transfer-learning dual-branch systems, a multi-task dual system and two MedViT V2 transformers. Custom CNNs use Adam (lr 1e-3, batch 32) with early stopping on validation loss; dual systems train in phases, frozen backbone first, fine-tuning after. See section 03.'],
             el: ['Εκπαίδευση εννέα αρχιτεκτονικών', 'Custom CNN, ένα ensemble ένας-προς-έναν, δύο συστήματα δύο κλάδων με μεταφορά μάθησης, ένα πολυεργασιακό σύστημα δύο κλάδων και δύο transformers MedViT V2. Τα custom CNN εκπαιδεύονται με Adam (lr 1e-3, batch 32) και early stopping στο validation loss· τα συστήματα δύο κλάδων εκπαιδεύονται σε φάσεις, πρώτα με παγωμένο backbone και μετά με fine-tuning. Δες την ενότητα 03.'],
             specs: ['Adam / AdamW', 'batch 32', 'early stopping', 'phased fine-tuning'] } },
      { id: 'e503',  x: 4, y: -0.62, kind: 'eval', en: ['503-image subset', 'cross-model comparison'], el: ['Υποσύνολο 503', 'σύγκριση όλων των μοντέλων'],
        d: { en: ['Common 503-image evaluation subset', 'Only tumour-positive test images (glioma 162, meningioma 165, pituitary 176), so that all nine models, including the 3-Class CNN and the Ensemble, answer the same question on the same images. Accuracy is reported with 95% Wilson score intervals.'],
             el: ['Κοινό υποσύνολο αξιολόγησης 503 εικόνων', 'Μόνο εικόνες ελέγχου με όγκο (γλοίωμα 162, μηνιγγίωμα 165, υπόφυση 176), ώστε και τα εννέα μοντέλα, μαζί με το 3-Class CNN και το Ensemble, να απαντούν στο ίδιο ερώτημα για τις ίδιες εικόνες. Η ακρίβεια δίνεται με διαστήματα Wilson 95%.'],
             specs: ['G 162 · M 165 · P 176', 'Wilson 95% CI', 'macro F1 · AUC'] } },
      { id: 'e703',  x: 4, y: 0.62, kind: 'eval', en: ['703-image test set', 'full 4-class evaluation'], el: ['Σύνολο 703', 'πλήρης αξιολόγηση 4 κλάσεων'],
        d: { en: ['Full four-class evaluation', 'All 703 test images including 200 healthy scans, for the three models that predict all four classes: 4-Class CNN, MedViT V2 Base and MedViT V2 Dual. Per-class precision, recall and F1.'],
             el: ['Πλήρης αξιολόγηση τεσσάρων κλάσεων', 'Και οι 703 εικόνες ελέγχου, μαζί με 200 υγιείς, για τα τρία μοντέλα που προβλέπουν και τις τέσσερις κλάσεις: 4-Class CNN, MedViT V2 Base και MedViT V2 Dual. Precision, recall και F1 ανά κλάση.'],
             specs: ['+200 no tumour', 'per-class F1', 'confusion matrix'] } },
      { id: 'stats', x: 5, y: -0.62, kind: 'eval', en: ['Statistics', 'Friedman · Nemenyi · Wilcoxon'], el: ['Στατιστική', 'Friedman · Nemenyi · Wilcoxon'],
        d: { en: ['Non-parametric significance testing', 'Following Demšar (2006): Friedman omnibus test, Nemenyi critical difference, pairwise Wilcoxon signed-rank tests with Bonferroni correction, Cohen\'s d effect sizes and 5-fold stratified cross-validation for stability. See section 05.'],
             el: ['Μη παραμετρικός έλεγχος σημαντικότητας', 'Κατά Demšar (2006): συνολικός έλεγχος Friedman, κρίσιμη διαφορά Nemenyi, ανά ζεύγη έλεγχοι Wilcoxon με διόρθωση Bonferroni, μεγέθη επίδρασης Cohen\'s d και 5-fold στρωματοποιημένο cross-validation για τη σταθερότητα. Δες την ενότητα 05.'],
             specs: ['χ²F = 164.45', 'CD = 0.536', '36 pairs', 'B = 10,000 bootstrap'] } },
      { id: 'xai',   x: 5, y: 0.62, kind: 'eval', en: ['Explainable AI', 'Grad-CAM · SHAP · IG · Occlusion'], el: ['Επεξηγήσιμη ΤΝ', 'Grad-CAM · SHAP · IG · Occlusion'],
        d: { en: ['Four complementary XAI methods', 'Gradient-based (Grad-CAM), game-theoretic (SHAP GradientExplainer), path-integral (Integrated Gradients, 50 steps) and perturbation-based (occlusion, 28 × 28 window, stride 7). Applied to one median-confidence, correctly classified sample per class. See section 06.'],
             el: ['Τέσσερις συμπληρωματικές μέθοδοι XAI', 'Με κλίσεις (Grad-CAM), θεωρία παιγνίων (SHAP GradientExplainer), ολοκλήρωμα διαδρομής (Integrated Gradients, 50 βήματα) και διαταραχή (occlusion, παράθυρο 28 × 28, βήμα 7). Εφαρμόζονται σε ένα σωστά ταξινομημένο δείγμα διάμεσης βεβαιότητας ανά κλάση. Δες την ενότητα 06.'],
             specs: ['Grad-CAM', 'SHAP', 'IG · 50 steps', 'occlusion 28/7'] } },
      { id: 'find',  x: 6, y: 0, kind: 'eval', en: ['Findings', 'tiers · paradox · hybrid'], el: ['Ευρήματα', 'κλιμάκια · παράδοξο · υβριδικό'],
        d: { en: ['What comes out', 'Three performance tiers, a statistically justified hybrid CNN–Transformer, practical equivalence among the top models, gradient saturation in transformers, and the Meningioma Paradox. See sections 07–09.'],
             el: ['Τι προκύπτει', 'Τρία κλιμάκια απόδοσης, ένα στατιστικά τεκμηριωμένο υβριδικό CNN–Transformer, πρακτική ισοδυναμία των κορυφαίων μοντέλων, κορεσμός κλίσεων στους transformers και το Παράδοξο του Μηνιγγιώματος. Δες τις ενότητες 07–09.'],
             specs: ['3 tiers', 'Meningioma Paradox'] } },
    ],
    edges: [['data', 'prep'], ['prep', 'split'], ['split', 'train'], ['train', 'e503'], ['train', 'e703'], ['e503', 'stats'], ['e703', 'stats'], ['e503', 'xai'], ['e703', 'xai'], ['stats', 'find'], ['xai', 'find']],
  };


  /* ───────── Clinical app flow ───────── */
  const APP = {
    nodes: [
      { id: 'up', x: 0, y: 0, kind: 'train', en: ['Upload MRI', 'PNG · JPG · BMP'], el: ['Ανέβασμα MRI', 'PNG · JPG · BMP'],
        d: { en: ['Upload one MRI slice', 'The clinician drops in a single slice. It is converted to RGB, resized to 224 × 224 and normalised exactly as during training, and a preview is shown beside the uploader.'],
             el: ['Ανέβασμα μίας τομής MRI', 'Ο γιατρός ανεβάζει μία τομή. Μετατρέπεται σε RGB, αλλάζει μέγεθος σε 224 × 224 και κανονικοποιείται ακριβώς όπως στην εκπαίδευση, και εμφανίζεται προεπισκόπηση δίπλα στο πεδίο ανεβάσματος.'],
             specs: ['224×224×3', '[0, 1]'] } },
      { id: 'models', x: 1, y: 0, kind: 'train', en: ['7 trained models', 'loaded once, cached'], el: ['7 εκπαιδευμένα μοντέλα', 'φορτώνονται μία φορά'],
        d: { en: ['Seven models in parallel', 'Binary CNN, 4-Class CNN, 3-Class CNN, the 1-vs-1 Ensemble (three binary CNNs), ResNet-50 Dual, EfficientNet-B3 Dual and the Multi-Dual System are loaded from the model directory and kept in the session. The sidebar shows which loaded successfully.'],
             el: ['Επτά μοντέλα παράλληλα', 'Τα Binary CNN, 4-Class CNN, 3-Class CNN, το 1-vs-1 Ensemble (τρία δυαδικά CNN), τα ResNet-50 Dual, EfficientNet-B3 Dual και το Multi-Dual System φορτώνονται από τον φάκελο μοντέλων και κρατιούνται στη συνεδρία. Η πλευρική στήλη δείχνει ποια φορτώθηκαν.'],
             specs: ['Keras .h5', 'session cache'] } },
      { id: 'pred', x: 2, y: 0, kind: 'eval', en: ['Per-model prediction', 'class + probabilities'], el: ['Πρόβλεψη ανά μοντέλο', 'κλάση + πιθανότητες'],
        d: { en: ['Every model answers', 'Each model returns its predicted class and full probability vector. The Ensemble reports its three pairwise votes; the Binary CNN answers only tumour versus no tumour.'],
             el: ['Κάθε μοντέλο απαντά', 'Κάθε μοντέλο επιστρέφει την προβλεπόμενη κλάση και ολόκληρο το διάνυσμα πιθανοτήτων. Το Ensemble δείχνει τις τρεις ψήφους των ζευγών· το Binary CNN απαντά μόνο όγκος ή όχι.'],
             specs: ['softmax', 'pairwise votes'] } },
      { id: 'cons', x: 3, y: -0.62, kind: 'eval', en: ['Consensus diagnosis', 'majority vote'], el: ['Κοινή διάγνωση', 'πλειοψηφία'],
        d: { en: ['Consensus across models', 'The multiclass models vote; the majority class becomes the consensus diagnosis, shown with the number of models in agreement and their mean confidence. Disagreement is itself a signal for the clinician.'],
             el: ['Κοινή απόφαση των μοντέλων', 'Τα πολυκλασικά μοντέλα ψηφίζουν· η κλάση της πλειοψηφίας γίνεται η κοινή διάγνωση, μαζί με το πόσα μοντέλα συμφωνούν και τη μέση βεβαιότητά τους. Η διαφωνία είναι από μόνη της σήμα για τον γιατρό.'],
             specs: ['votes x / y', 'mean confidence'] } },
      { id: 'xai', x: 3, y: 0.62, kind: 'eval', en: ['XAI explanations', 'Grad-CAM · IG · SHAP · Occl.'], el: ['Εξηγήσεις XAI', 'Grad-CAM · IG · SHAP · Occl.'],
        d: { en: ['Explanations on demand', 'Each XAI method can be switched on in the sidebar. Heatmaps are overlaid on the uploaded slice for every model, in separate tabs, so the clinician can see where each network looked.'],
             el: ['Εξηγήσεις κατ\' απαίτηση', 'Κάθε μέθοδος XAI ενεργοποιείται από την πλευρική στήλη. Οι χάρτες θερμότητας υπερτίθενται στην τομή για κάθε μοντέλο, σε ξεχωριστές καρτέλες, ώστε ο γιατρός να βλέπει πού κοίταξε κάθε δίκτυο.'],
             specs: ['Grad-CAM', 'Integrated Gradients', 'SHAP', 'Occlusion'] } },
      { id: 'pdf', x: 4, y: 0, kind: 'eval', en: ['PDF report', 'download'], el: ['Αναφορά PDF', 'λήψη'],
        d: { en: ['Clinical report', 'One click generates a PDF with the consensus, per-model results and explanation figures, ready to attach to a case file.'],
             el: ['Κλινική αναφορά', 'Με ένα κλικ δημιουργείται PDF με την κοινή διάγνωση, τα αποτελέσματα ανά μοντέλο και τα σχήματα εξήγησης, έτοιμο να μπει στον φάκελο του περιστατικού.'],
             specs: ['ReportLab', 'A4'] } },
    ],
    edges: [['up', 'models'], ['models', 'pred'], ['pred', 'cons'], ['pred', 'xai'], ['cons', 'pdf'], ['xai', 'pdf']],
  };

  /* ───────── Architectures ───────── */
  // Reusable block descriptions.
  const B = {
    input: { kind: 'in', en: ['Input', '224 × 224 × 3'], el: ['Είσοδος', '224 × 224 × 3'],
      d: { en: 'An MRI slice resized to 224 × 224 pixels with three channels, values scaled to [0, 1].', el: 'Μια τομή MRI σε 224 × 224 εικονοστοιχεία με τρία κανάλια, τιμές στο [0, 1].' } },
    conv: (f, drop) => ({ kind: 'conv', en: [`Conv block ${f}`, `2×Conv3×3 · BN · Pool · Drop ${drop}`], el: [`Συνελικτικό μπλοκ ${f}`, `2×Conv3×3 · BN · Pool · Drop ${drop}`],
      d: { en: `Two 3 × 3 convolutions with ${f} filters and ReLU, each followed by batch normalisation, then 2 × 2 max pooling and dropout ${drop}. Spatial resolution halves, channels grow.`,
           el: `Δύο συνελίξεις 3 × 3 με ${f} φίλτρα και ReLU, η καθεμία με batch normalisation, έπειτα 2 × 2 max pooling και dropout ${drop}. Η χωρική ανάλυση υποδιπλασιάζεται, τα κανάλια αυξάνονται.` } }),
    convStack: { kind: 'conv', en: ['4 conv blocks', '32 → 64 → 128 → 256'], el: ['4 συνελικτικά μπλοκ', '32 → 64 → 128 → 256'],
      d: { en: 'The custom CNN backbone: four blocks of two 3 × 3 convolutions with batch normalisation, max pooling and dropout, from 32 to 256 filters.', el: 'Το custom CNN backbone: τέσσερα μπλοκ με δύο συνελίξεις 3 × 3, batch normalisation, max pooling και dropout, από 32 έως 256 φίλτρα.' } },
    gap: { kind: 'dense', en: ['Global avg pool', '256-d'], el: ['Global avg pool', '256-d'],
      d: { en: 'Averages each feature map to a single number. Far fewer parameters than flattening, which is why the custom CNNs stay near 1.3 M.', el: 'Μετατρέπει κάθε χάρτη χαρακτηριστικών σε έναν αριθμό. Πολύ λιγότερες παράμετροι από το flatten, γι\' αυτό τα custom CNN μένουν κοντά στα 1,3 M.' } },
    dense: (u, drop) => ({ kind: 'dense', en: [`Dense ${u}`, `ReLU · BN · Drop ${drop}`], el: [`Dense ${u}`, `ReLU · BN · Drop ${drop}`],
      d: { en: `Fully connected layer with ${u} units, ReLU, batch normalisation and dropout ${drop}.`, el: `Πλήρως συνδεδεμένο επίπεδο ${u} μονάδων με ReLU, batch normalisation και dropout ${drop}.` } }),
    soft: (k, lbl) => ({ kind: 'out', en: [`Softmax ${k}`, lbl.en], el: [`Softmax ${k}`, lbl.el],
      d: { en: `Output probabilities over ${k} classes: ${lbl.en}.`, el: `Πιθανότητες εξόδου για ${k} κλάσεις: ${lbl.el}.` } }),
  };
  const L4 = { en: 'G · M · N · P', el: 'Γ · Μ · Χ · Υ' };
  const L3 = { en: 'G · M · P', el: 'Γ · Μ · Υ' };

  function seq(list) {
    // Lay out a straight chain on lane 0.
    return { nodes: list.map((b, i) => Object.assign({ id: 'n' + i, x: i, y: 0 }, b)), edges: list.slice(1).map((_, i) => ['n' + i, 'n' + (i + 1)]) };
  }

  const ARCH = [
    {
      id: 'binary', model: 1,
      tag: { en: 'Custom CNN · 2 classes', el: 'Custom CNN · 2 κλάσεις' },
      desc: { en: 'A tumour detector: does the slice contain a tumour at all? Four convolutional blocks with increasingly strong dropout feed a sigmoid output.', el: 'Ανιχνευτής όγκου: περιέχει η τομή όγκο ή όχι; Τέσσερα συνελικτικά μπλοκ με ολοένα ισχυρότερο dropout καταλήγουν σε έξοδο sigmoid.' },
      graph: seq([B.input, B.conv(32, .25), B.conv(64, .35), B.conv(128, .4), B.conv(256, .5), B.dense(512, .5), B.dense(256, .5),
        { kind: 'out', en: ['Sigmoid', 'tumour / no tumour'], el: ['Sigmoid', 'όγκος / χωρίς όγκο'], d: { en: 'A single probability: tumour (1) versus no tumour (0).', el: 'Μία πιθανότητα: όγκος (1) ή χωρίς όγκο (0).' } }]),
    },
    {
      id: 'cnn4', model: 2,
      tag: { en: 'Custom CNN · 4 classes', el: 'Custom CNN · 4 κλάσεις' },
      desc: { en: 'The workhorse of the thesis: 1.28 M parameters, four classes, and the CNN branch that MedViT V2 Dual later reuses.', el: 'Το «άλογο εργασίας» της εργασίας: 1,28 M παράμετροι, τέσσερις κλάσεις, και ο κλάδος CNN που επαναχρησιμοποιεί αργότερα το MedViT V2 Dual.' },
      graph: seq([B.input, B.conv(32, .25), B.conv(64, .25), B.conv(128, .3), B.conv(256, .3), B.gap, B.dense(256, .5), B.dense(128, .4), B.soft(4, L4)]),
    },
    {
      id: 'cnn3', model: 3,
      tag: { en: 'Custom CNN · 3 tumour types', el: 'Custom CNN · 3 τύποι όγκου' },
      desc: { en: 'Same backbone as the 4-Class CNN, trained only on tumour images. It answers "which tumour?", assuming a tumour is present.', el: 'Ίδιο backbone με το 4-Class CNN, εκπαιδευμένο μόνο σε εικόνες με όγκο. Απαντά «ποιος όγκος;», με την παραδοχή ότι υπάρχει όγκος.' },
      graph: seq([B.input, B.conv(32, .25), B.conv(64, .25), B.conv(128, .3), B.conv(256, .3), B.gap, B.dense(256, .5), B.dense(128, .4), B.soft(3, L3)]),
    },
    {
      id: 'ensemble', model: 4,
      tag: { en: 'Three binary CNNs · voting', el: 'Τρία δυαδικά CNN · ψηφοφορία' },
      desc: { en: 'Splits the three-class problem into three pairwise duels. Each binary CNN specialises in one boundary; their probabilities are combined by voting.', el: 'Σπάει το πρόβλημα τριών κλάσεων σε τρεις μονομαχίες ανά ζεύγη. Κάθε δυαδικό CNN ειδικεύεται σε ένα όριο· οι πιθανότητές τους συνδυάζονται με ψηφοφορία.' },
      graph: {
        nodes: [
          Object.assign({ id: 'in', x: 0, y: 0 }, B.input),
          { id: 'gm', x: 1.3, y: -1, kind: 'conv', en: ['CNN: G vs M', 'binary'], el: ['CNN: Γ έναντι Μ', 'δυαδικό'], d: { en: 'Binary CNN trained only on glioma and meningioma images.', el: 'Δυαδικό CNN εκπαιδευμένο μόνο σε γλοιώματα και μηνιγγιώματα.' } },
          { id: 'gp', x: 1.3, y: 0, kind: 'conv', en: ['CNN: G vs P', 'binary'], el: ['CNN: Γ έναντι Υ', 'δυαδικό'], d: { en: 'Binary CNN trained only on glioma and pituitary images.', el: 'Δυαδικό CNN εκπαιδευμένο μόνο σε γλοιώματα και όγκους υπόφυσης.' } },
          { id: 'mp', x: 1.3, y: 1, kind: 'conv', en: ['CNN: M vs P', 'binary'], el: ['CNN: Μ έναντι Υ', 'δυαδικό'], d: { en: 'Binary CNN trained only on meningioma and pituitary images.', el: 'Δυαδικό CNN εκπαιδευμένο μόνο σε μηνιγγιώματα και όγκους υπόφυσης.' } },
          { id: 'vote', x: 2.6, y: 0, kind: 'fuse', en: ['Probability voting', 'sum per class'], el: ['Ψηφοφορία πιθανοτήτων', 'άθροισμα ανά κλάση'], d: { en: 'Each duel adds its probabilities to the two classes it knows; the class with the highest total wins.', el: 'Κάθε μονομαχία προσθέτει τις πιθανότητές της στις δύο κλάσεις που γνωρίζει· κερδίζει η κλάση με το μεγαλύτερο άθροισμα.' } },
          Object.assign({ id: 'out', x: 3.9, y: 0 }, B.soft(3, L3), { en: ['Argmax', L3.en], el: ['Argmax', L3.el] }),
        ],
        edges: [['in', 'gm'], ['in', 'gp'], ['in', 'mp'], ['gm', 'vote'], ['gp', 'vote'], ['mp', 'vote'], ['vote', 'out']],
      },
    },
    {
      id: 'resnet', model: 5,
      tag: { en: 'Transfer learning · dual branch', el: 'Μεταφορά μάθησης · δύο κλάδοι' },
      desc: { en: 'A custom CNN branch beside an ImageNet-pretrained ResNet-50. The pretrained branch is frozen first, then partially unfrozen across three training phases. The ImageNet features transfer poorly to MRI: lowest tier.', el: 'Ένας κλάδος custom CNN δίπλα σε ένα ResNet-50 προεκπαιδευμένο στο ImageNet. Ο προεκπαιδευμένος κλάδος είναι αρχικά παγωμένος και ξεπαγώνει σταδιακά σε τρεις φάσεις εκπαίδευσης. Τα χαρακτηριστικά του ImageNet μεταφέρονται άσχημα στο MRI: χαμηλότερο κλιμάκιο.' },
      graph: dualTL('ResNet-50', 'ImageNet · 3 phases', 'Residual network with 50 layers pretrained on ImageNet natural images. Frozen in phase 1, top layers unfrozen for fine-tuning later.', 'Υπολειμματικό δίκτυο 50 επιπέδων προεκπαιδευμένο σε φυσικές εικόνες του ImageNet. Παγωμένο στη φάση 1, τα ανώτερα επίπεδα ξεπαγώνουν αργότερα για fine-tuning.'),
    },
    {
      id: 'efnet', model: 6,
      tag: { en: 'Transfer learning · dual branch', el: 'Μεταφορά μάθησης · δύο κλάδοι' },
      desc: { en: 'Same design with EfficientNet-B3 as the pretrained branch. Better than ResNet-50 at half the size, still ten points behind the custom CNNs.', el: 'Ίδιος σχεδιασμός με EfficientNet-B3 ως προεκπαιδευμένο κλάδο. Καλύτερο από το ResNet-50 με το μισό μέγεθος, αλλά ακόμα δέκα μονάδες πίσω από τα custom CNN.' },
      graph: dualTL('EfficientNet-B3', 'ImageNet · 3 phases', 'Compound-scaled convolutional network pretrained on ImageNet. Frozen first, then fine-tuned in later phases.', 'Συνελικτικό δίκτυο με σύνθετη κλιμάκωση, προεκπαιδευμένο στο ImageNet. Αρχικά παγωμένο, έπειτα fine-tuning στις επόμενες φάσεις.'),
    },
    {
      id: 'multidual', model: 7,
      tag: { en: 'Shared backbone · two heads', el: 'Κοινό backbone · δύο κεφαλές' },
      desc: { en: 'One CNN backbone, two task heads: a 3-class tumour-type head trained with a masked loss (healthy images ignored) and a 4-class head. A fusion layer combines both.', el: 'Ένα CNN backbone, δύο κεφαλές: μία 3 κλάσεων για τον τύπο όγκου με masked loss (οι υγιείς εικόνες αγνοούνται) και μία 4 κλάσεων. Ένα επίπεδο σύντηξης συνδυάζει και τις δύο.' },
      graph: {
        nodes: [
          Object.assign({ id: 'in', x: 0, y: 0 }, B.input),
          Object.assign({ id: 'bb', x: 1, y: 0 }, B.convStack),
          { id: 'fd', x: 2, y: 0, kind: 'dense', en: ['Shared Dense 512', 'feature vector'], el: ['Κοινό Dense 512', 'διάνυσμα χαρακτηριστικών'], d: { en: 'Shared 512-d representation consumed by both heads.', el: 'Κοινή αναπαράσταση 512 διαστάσεων που τροφοδοτεί και τις δύο κεφαλές.' } },
          { id: 'h1', x: 3.2, y: -0.75, kind: 'dense', en: ['Head 1: 256 → 128', '3-class · masked loss'], el: ['Κεφαλή 1: 256 → 128', '3 κλάσεις · masked loss'], d: { en: 'Tumour-type head. Its loss is masked so healthy images contribute no gradient; this extra optimisation constraint is what costs the system accuracy.', el: 'Κεφαλή τύπου όγκου. Το loss της είναι masked, ώστε οι υγιείς εικόνες να μη δίνουν κλίση· αυτός ο επιπλέον περιορισμός βελτιστοποίησης είναι που κοστίζει ακρίβεια στο σύστημα.' } },
          { id: 'h2', x: 3.2, y: 0.75, kind: 'dense', en: ['Head 2: 256 → 128', '4-class'], el: ['Κεφαλή 2: 256 → 128', '4 κλάσεις'], d: { en: 'Four-class head trained with standard cross-entropy.', el: 'Κεφαλή τεσσάρων κλάσεων με κανονικό cross-entropy.' } },
          { id: 'fu', x: 4.4, y: 0, kind: 'fuse', en: ['Fusion', 'concat → Dense 128'], el: ['Σύντηξη', 'concat → Dense 128'], d: { en: 'Concatenates the two heads\' features and learns how to weigh them.', el: 'Ενώνει τα χαρακτηριστικά των δύο κεφαλών και μαθαίνει πώς να τα σταθμίζει.' } },
          Object.assign({ id: 'out', x: 5.4, y: 0 }, B.soft(4, L4)),
        ],
        edges: [['in', 'bb'], ['bb', 'fd'], ['fd', 'h1'], ['fd', 'h2'], ['h1', 'fu'], ['h2', 'fu'], ['fu', 'out']],
      },
    },
    {
      id: 'medvit', model: 8,
      tag: { en: 'Hybrid transformer · single branch', el: 'Υβριδικός transformer · ένας κλάδος' },
      desc: { en: 'MedViT V2, a medical vision transformer that mixes local and global attention in every stage. 38× the parameters of the 4-Class CNN, yet five points less accurate on its own.', el: 'Το MedViT V2, ένας ιατρικός vision transformer που αναμειγνύει τοπική και καθολική προσοχή σε κάθε στάδιο. 38 φορές τις παραμέτρους του 4-Class CNN, κι όμως μόνο του πέντε μονάδες λιγότερο ακριβές.' },
      graph: seq([B.input,
        { kind: 'conv', en: ['Conv stem', '4 conv layers'], el: ['Conv stem', '4 συνελικτικά επίπεδα'], d: { en: 'Four convolutional layers that turn pixels into a lower-resolution feature map before any attention.', el: 'Τέσσερα συνελικτικά επίπεδα που μετατρέπουν τα εικονοστοιχεία σε χάρτη χαρακτηριστικών χαμηλότερης ανάλυσης πριν από κάθε μηχανισμό προσοχής.' } },
        { kind: 'attn', en: ['LFP block × N', 'DiNA + LFFN'], el: ['Μπλοκ LFP × N', 'DiNA + LFFN'], d: { en: 'Local Feature Perception: Dilated Neighbourhood Attention looks at a dilated 7 × 7 neighbourhood around each position, followed by a local feed-forward network.', el: 'Local Feature Perception: η Dilated Neighbourhood Attention κοιτάζει μια αραιωμένη γειτονιά 7 × 7 γύρω από κάθε θέση, και ακολουθεί τοπικό δίκτυο πρόσθιας τροφοδότησης.' } },
        { kind: 'attn', en: ['GFP block × 1', 'E-MHSA + MHCA + KAN'], el: ['Μπλοκ GFP × 1', 'E-MHSA + MHCA + KAN'], d: { en: 'Global Feature Perception: efficient multi-head self-attention over the whole image, multi-head convolutional attention, and a Kolmogorov–Arnold network layer.', el: 'Global Feature Perception: αποδοτική multi-head self-attention σε όλη την εικόνα, multi-head convolutional attention, και ένα επίπεδο δικτύου Kolmogorov–Arnold.' } },
        { kind: 'attn', en: ['× 4 stages', 'hierarchical'], el: ['× 4 στάδια', 'ιεραρχικά'], d: { en: 'The (LFP × N + GFP × 1) pattern repeats over four stages of decreasing resolution. Grad-CAM for MedViT is taken from the s4_l1_gfp0 projection, a 7 × 7 grid.', el: 'Το μοτίβο (LFP × N + GFP × 1) επαναλαμβάνεται σε τέσσερα στάδια με φθίνουσα ανάλυση. Το Grad-CAM του MedViT λαμβάνεται από την προβολή s4_l1_gfp0, ένα πλέγμα 7 × 7.' } },
        { kind: 'dense', en: ['LayerNorm · GAP', 'head_ln'], el: ['LayerNorm · GAP', 'head_ln'], d: { en: 'Layer normalisation and global average pooling to a single feature vector.', el: 'Κανονικοποίηση επιπέδου και global average pooling σε ένα διάνυσμα χαρακτηριστικών.' } },
        B.soft(4, L4)]),
    },
    {
      id: 'medvitdual', model: 9,
      tag: { en: 'Proposed · CNN + Transformer fusion', el: 'Προτεινόμενο · σύντηξη CNN + Transformer' },
      desc: { en: 'The proposed hybrid. Both pretrained models see the same image; their feature vectors are concatenated and a small MLP learns to combine local CNN evidence with global transformer context. Trained in two phases.', el: 'Το προτεινόμενο υβριδικό. Και τα δύο προεκπαιδευμένα μοντέλα βλέπουν την ίδια εικόνα· τα διανύσματα χαρακτηριστικών τους ενώνονται και ένα μικρό MLP μαθαίνει να συνδυάζει την τοπική πληροφορία του CNN με το καθολικό πλαίσιο του transformer. Εκπαιδεύεται σε δύο φάσεις.' },
      graph: {
        nodes: [
          Object.assign({ id: 'in', x: 0, y: 0 }, B.input),
          { id: 'vit', x: 1.2, y: -0.75, kind: 'attn', en: ['MedViT V2 branch', 'head_ln → GAP'], el: ['Κλάδος MedViT V2', 'head_ln → GAP'], d: { en: 'The pretrained MedViT V2 up to its final LayerNorm, pooled to a vector: global, long-range context.', el: 'Το προεκπαιδευμένο MedViT V2 μέχρι το τελικό LayerNorm, συμπιεσμένο σε διάνυσμα: καθολικό πλαίσιο μεγάλης εμβέλειας.' } },
          { id: 'cnn', x: 1.2, y: 0.75, kind: 'conv', en: ['4-Class CNN branch', 'dense2 · 128-d'], el: ['Κλάδος 4-Class CNN', 'dense2 · 128-d'], d: { en: 'The pretrained 4-Class CNN up to its 128-unit dense layer: local, lesion-focused features.', el: 'Το προεκπαιδευμένο 4-Class CNN μέχρι το dense επίπεδο 128 μονάδων: τοπικά χαρακτηριστικά εστιασμένα στη βλάβη.' } },
          { id: 'cat', x: 2.4, y: 0, kind: 'fuse', en: ['Concatenate', 'BatchNorm'], el: ['Concatenate', 'BatchNorm'], d: { en: 'Both vectors are joined and normalised so neither branch dominates by scale.', el: 'Τα δύο διανύσματα ενώνονται και κανονικοποιούνται, ώστε κανένας κλάδος να μην κυριαρχεί λόγω κλίμακας.' } },
          Object.assign({ id: 'd1', x: 3.4, y: 0 }, B.dense(512, .4)),
          Object.assign({ id: 'd2', x: 4.4, y: 0 }, B.dense(256, .3)),
          Object.assign({ id: 'out', x: 5.4, y: 0 }, B.soft(4, L4)),
          { id: 'p1', x: 1.2, y: 1.75, kind: 'note', en: ['Phase 1', 'branches frozen · AdamW 1e-3'], el: ['Φάση 1', 'κλάδοι παγωμένοι · AdamW 1e-3'], d: { en: 'Phase 1: only the fusion head trains (AdamW, lr 1e-3, up to 40 epochs, early-stopping patience 10).', el: 'Φάση 1: εκπαιδεύεται μόνο η κεφαλή σύντηξης (AdamW, lr 1e-3, έως 40 εποχές, early stopping με υπομονή 10).' } },
          { id: 'p2', x: 3.4, y: 1.75, kind: 'note', en: ['Phase 2', 'all unfrozen · AdamW 1e-4'], el: ['Φάση 2', 'όλα ξεπαγωμένα · AdamW 1e-4'], d: { en: 'Phase 2: both branches unfrozen and fine-tuned end-to-end at a lower rate (AdamW, lr 1e-4, up to 60 epochs, patience 15).', el: 'Φάση 2: οι δύο κλάδοι ξεπαγώνουν και γίνεται fine-tuning από άκρη σε άκρη με μικρότερο ρυθμό (AdamW, lr 1e-4, έως 60 εποχές, υπομονή 15).' } },
        ],
        edges: [['in', 'vit'], ['in', 'cnn'], ['vit', 'cat'], ['cnn', 'cat'], ['cat', 'd1'], ['d1', 'd2'], ['d2', 'out'], ['p1', 'p2', 'dash']],
      },
    },
  ];

  function dualTL(name, sub, dEn, dEl) {
    return {
      nodes: [
        Object.assign({ id: 'in', x: 0, y: 0 }, B.input),
        Object.assign({ id: 'cnn', x: 1.2, y: -0.75 }, B.convStack, { en: ['Custom CNN branch', '32 → 256 · GAP'], el: ['Κλάδος custom CNN', '32 → 256 · GAP'] }),
        { id: 'tl', x: 1.2, y: 0.75, kind: 'attn', en: [name, sub], el: [name, sub.replace('phases', 'φάσεις')], d: { en: dEn, el: dEl } },
        { id: 'cd', x: 2.4, y: -0.75, kind: 'dense', en: ['Dense', 'branch features'], el: ['Dense', 'χαρακτηριστικά κλάδου'], d: { en: 'Dense layers project the CNN features to a compact vector.', el: 'Πυκνά επίπεδα προβάλλουν τα χαρακτηριστικά του CNN σε συμπαγές διάνυσμα.' } },
        { id: 'td', x: 2.4, y: 0.75, kind: 'dense', en: ['GAP · Dense 128', 'branch features'], el: ['GAP · Dense 128', 'χαρακτηριστικά κλάδου'], d: { en: 'Global pooling and a 128-unit dense layer on top of the pretrained backbone.', el: 'Global pooling και dense επίπεδο 128 μονάδων πάνω από το προεκπαιδευμένο backbone.' } },
        { id: 'fu', x: 3.6, y: 0, kind: 'fuse', en: ['Fusion', 'concat · BN · dropout'], el: ['Σύντηξη', 'concat · BN · dropout'], d: { en: 'Concatenation of both branches, batch normalisation and dropout, then dense layers.', el: 'Ένωση των δύο κλάδων, batch normalisation και dropout, έπειτα dense επίπεδα.' } },
        Object.assign({ id: 'out', x: 4.7, y: 0 }, B.soft(4, L4)),
      ],
      edges: [['in', 'cnn'], ['in', 'tl'], ['cnn', 'cd'], ['tl', 'td'], ['cd', 'fu'], ['td', 'fu'], ['fu', 'out']],
    };
  }

  /* ───────── XAI ───────── */
  const XAI_METHODS = [
    { id: 'gradcam', name: 'Grad-CAM',
      en: '<b>Grad-CAM</b> weights the last convolutional feature maps by the gradient of the predicted class and overlays the result. Red regions push the decision most strongly. Coarse, but fast and class-specific.',
      el: '<b>Grad-CAM</b>: σταθμίζει τους τελευταίους συνελικτικούς χάρτες χαρακτηριστικών με την κλίση της προβλεπόμενης κλάσης και υπερθέτει το αποτέλεσμα. Οι κόκκινες περιοχές ωθούν περισσότερο την απόφαση. Χονδροειδές, αλλά γρήγορο και ειδικό ανά κλάση.' },
    { id: 'shap', name: 'SHAP',
      en: '<b>SHAP</b> (GradientExplainer) estimates each pixel\'s Shapley value: its fair share of the prediction, relative to 20 background images. Bright speckles mark pixels that raise the predicted class.',
      el: '<b>SHAP</b> (GradientExplainer): εκτιμά την τιμή Shapley κάθε εικονοστοιχείου, δηλαδή το «δίκαιο μερίδιό» του στην πρόβλεψη, σε σχέση με 20 εικόνες αναφοράς. Οι φωτεινές κουκκίδες δείχνουν εικονοστοιχεία που ενισχύουν την προβλεπόμενη κλάση.' },
    { id: 'ig', name: 'Integrated Gradients',
      en: '<b>Integrated Gradients</b> accumulates gradients along a straight path from a black image to the input (50 steps). It satisfies completeness: attributions sum to the change in the model\'s output.',
      el: '<b>Integrated Gradients</b>: αθροίζει κλίσεις κατά μήκος μιας ευθείας διαδρομής από μια μαύρη εικόνα έως την είσοδο (50 βήματα). Ικανοποιεί την πληρότητα: οι αποδόσεις αθροίζονται στη μεταβολή της εξόδου του μοντέλου.' },
    { id: 'occlusion', name: 'Occlusion',
      en: '<b>Occlusion</b> slides a grey 28 × 28 patch across the image (stride 7) and records how much the confidence drops. No gradients needed, so it works the same for every architecture. Red means "covering this hurts".',
      el: '<b>Occlusion</b>: σέρνει ένα γκρι τετράγωνο 28 × 28 πάνω στην εικόνα (βήμα 7) και καταγράφει πόσο πέφτει η βεβαιότητα. Δεν χρειάζεται κλίσεις, άρα λειτουργεί ίδια για κάθε αρχιτεκτονική. Κόκκινο σημαίνει «αν το καλύψεις, πονάει».' },
  ];

  const XAI_MODELS = [
    { id: 'binary',    name: 'Binary CNN' },
    { id: 'cnn4',      name: '4-Class CNN' },
    { id: 'cnn3',      name: '3-Class CNN' },
    { id: 'resnet',    name: 'ResNet-50 Dual' },
    { id: 'efnet',     name: 'EfficientNet-B3 Dual' },
    { id: 'multidual', name: 'Multi-Dual System' },
    { id: 'medvit',    name: 'MedViT V2' },
  ];

  const XAI_NOTES = {
    family: {
      binary:    { en: 'A two-class detector: its map shows evidence for "tumour" in general, not for a specific tumour type.', el: 'Ανιχνευτής δύο κλάσεων: ο χάρτης του δείχνει ενδείξεις για «όγκο» γενικά, όχι για συγκεκριμένο τύπο.' },
      cnn4:      { en: 'Custom CNNs produce spatially concentrated activations aligned with clinically plausible regions.', el: 'Τα custom CNN δίνουν χωρικά συγκεντρωμένες ενεργοποιήσεις, σε κλινικά εύλογες περιοχές.' },
      cnn3:      { en: 'Same backbone as the 4-Class CNN, trained only on tumours: it cannot explain a healthy scan.', el: 'Ίδιο backbone με το 4-Class CNN, εκπαιδευμένο μόνο σε όγκους: δεν μπορεί να εξηγήσει μια υγιή εικόνα.' },
      resnet:    { en: 'ImageNet-pretrained branch: attributions tend to be broader and less lesion-specific, in line with its lowest-tier accuracy.', el: 'Κλάδος προεκπαιδευμένος στο ImageNet: οι αποδόσεις τείνουν να είναι ευρύτερες και λιγότερο εστιασμένες στη βλάβη, σε συμφωνία με την ακρίβεια του χαμηλότερου κλιμακίου.' },
      efnet:     { en: 'ImageNet-pretrained branch, more focused than ResNet-50 but still less consistent than the custom CNNs.', el: 'Κλάδος προεκπαιδευμένος στο ImageNet, πιο εστιασμένος από το ResNet-50 αλλά λιγότερο συνεπής από τα custom CNN.' },
      multidual: { en: 'Shared CNN backbone with two heads. For pituitary tumours occlusion reaches a 100% confidence drop: a highly specialised sellar-region representation.', el: 'Κοινό CNN backbone με δύο κεφαλές. Στους όγκους υπόφυσης το occlusion φτάνει πτώση βεβαιότητας 100%: πολύ εξειδικευμένη αναπαράσταση της περιοχής του τουρκικού εφιππίου.' },
    },
    medvit: {
      gradcam:   { en: 'Taken from a 7 × 7 projection layer (s4_l1_gfp0), so the map is coarse and spreads well beyond the tumour: the transformer integrates long-range context.', el: 'Λαμβάνεται από επίπεδο προβολής 7 × 7 (s4_l1_gfp0), οπότε ο χάρτης είναι χονδροειδής και απλώνεται πολύ πέρα από τον όγκο: ο transformer ενσωματώνει πλαίσιο μεγάλης εμβέλειας.' },
      shap:      { en: 'Almost blank, although the model is correct at high confidence. This is gradient saturation: near the decision, the transformer\'s gradients vanish and gradient-based attribution has nothing to show.', el: 'Σχεδόν κενό, παρότι το μοντέλο είναι σωστό με υψηλή βεβαιότητα. Είναι ο κορεσμός κλίσεων: κοντά στην απόφαση οι κλίσεις του transformer μηδενίζονται και οι μέθοδοι κλίσεων δεν έχουν τι να δείξουν.' },
      ig:        { en: 'Almost blank for the same reason as SHAP: gradient saturation. A real interpretability limit of attention-based models at high confidence.', el: 'Σχεδόν κενό για τον ίδιο λόγο με το SHAP: κορεσμός κλίσεων. Ένα πραγματικό όριο ερμηνευσιμότητας των μοντέλων προσοχής σε υψηλή βεβαιότητα.' },
      occlusion: { en: 'Perturbation still works where gradients fail. Sensitivity is spread over a broader area than in CNNs, consistent with global attention.', el: 'Η διαταραχή λειτουργεί εκεί που αποτυγχάνουν οι κλίσεις. Η ευαισθησία απλώνεται σε ευρύτερη περιοχή απ\' ό,τι στα CNN, σε συμφωνία με την καθολική προσοχή.' },
    },
    cls: {
      glioma:     { en: 'Gliomas grow inside the brain tissue (intra-axial) with irregular margins.', el: 'Τα γλοιώματα αναπτύσσονται μέσα στον εγκεφαλικό ιστό (ενδοαξονικά) με ακανόνιστα όρια.' },
      meningioma: { en: 'Meningiomas arise from the meninges, at the brain\'s edge against the skull. Every model finds them here, yet this is the class they misclassify most: the Meningioma Paradox.', el: 'Τα μηνιγγιώματα προέρχονται από τις μήνιγγες, στο όριο του εγκεφάλου με το κρανίο. Κάθε μοντέλο τα βρίσκει εκεί, κι όμως είναι η κλάση με τα περισσότερα λάθη: το Παράδοξο του Μηνιγγιώματος.' },
      no_tumor:   { en: 'With no lesion to find, the maps show what the network uses as evidence of normal anatomy.', el: 'Χωρίς βλάβη να εντοπίσουν, οι χάρτες δείχνουν τι χρησιμοποιεί το δίκτυο ως ένδειξη φυσιολογικής ανατομίας.' },
      pituitary:  { en: 'Pituitary adenomas sit in the sellar region at the skull base; CNNs focus there consistently.', el: 'Τα αδενώματα υπόφυσης βρίσκονται στην περιοχή του τουρκικού εφιππίου, στη βάση του κρανίου· τα CNN εστιάζουν εκεί με συνέπεια.' },
    },
  };

  return { CLASSES, MODELS, F1, CM, BOOT, CV, PAIR, STATS, PIPELINE, APP, ARCH, XAI_METHODS, XAI_MODELS, XAI_NOTES };
})();
