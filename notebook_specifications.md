# Technical Specification
# Consumer Credit Scoring Model

metadata:
  template_id: "tech_spec_consumer_credit_scoring_v1"
  template_name: "Consumer Credit Scoring Model - Technical Specification"
  version: "1.0.0"
  parent_prd: "prd_consumer_credit_scoring_v1"
  created_date: "2025-10-29"
  author: "QCreate Team / QuantUniversity"
  
  # Paper sources for this template
  academic_sources:
    - title: "Credit Scoring and Its Applications"
      authors: "Thomas, L.C., Edelman, D.B., Crook, J.N."
      year: 2002
      publisher: "SIAM"
      relevant_chapters: "Chapter 2 (Statistical Methods), Chapter 5 (Logistic Regression)"
    
    - title: "The Pricing of Credit Risk"
      authors: "Merton, R.C."
      year: 1974
      journal: "Journal of Finance"
      relevance: "Theoretical foundation for default modeling"
  
  regulatory_sources:
    - title: "SR 11-7: Supervisory Guidance on Model Risk Management"
      issuer: "Federal Reserve / OCC"
      date: "2011-04-04"
      url: "https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm"
    
    - title: "Equal Credit Opportunity Act (ECOA)"
      regulation: "15 U.S.C. § 1691"
      regulation_b: "12 CFR Part 1002"

# ============================================================================
# SECTION 1: THEORETICAL FOUNDATION
# ============================================================================

theoretical_foundation:
  economic_theory: |
    Credit default risk fundamentally reflects borrower's ABILITY and WILLINGNESS to repay:
    
    1. ABILITY TO PAY (Capacity):
       - Income level (absolute debt service capacity)
       - Debt-to-income ratio (relative leverage)
       - Employment stability (income persistence)
    
    2. WILLINGNESS TO PAY (Character):
       - Credit score (payment history, track record)
       - Recent delinquencies (behavioral patterns)
    
    Framework: Merton's structural model (1974) where default occurs when 
    borrower's assets fall below liabilities. For consumer credit, we proxy:
    - Assets → Income stream (present value of future earnings)
    - Liabilities → Existing debt obligations + new loan
    - Default threshold → Point where debt service >income capacity
  
  statistical_methodology: |
    Logistic regression selected because:
    
    1. INTERPRETABILITY: Coefficients have clear economic meaning
       - β₁ (income) = marginal impact of $1K income on log-odds of default
       - exp(β₁) = odds ratio (multiplicative effect)
    
    2. PROBABILISTIC OUTPUT: Natural for binary outcomes (default / no default)
       - Outputs well-calibrated probabilities (can set decision thresholds)
    
    3. REGULATORY ACCEPTANCE: Widely used, understood by auditors/examiners
       - Not a "black box" (unlike neural networks)
       - SR 11-7 conceptual soundness easily demonstrated
    
    4. COMPUTATIONAL EFFICIENCY: Fast training and prediction
       - <100ms prediction time (real-time decisioning)
       - Stable numerical optimization (MLE converges reliably)
  
  alternative_methods_considered:
    - method: "Decision Trees / Random Forests"
      pros: "Non-linear relationships, feature interactions"
      cons: "Less interpretable, harder to explain for adverse actions"
      decision: "Rejected - ECOA requires explainability"
    
    - method: "Neural Networks"
      pros: "Can capture complex patterns"
      cons: "Black box, difficult to validate per SR 11-7"
      decision: "Rejected - Regulatory acceptance uncertain"
    
    - method: "Linear Discriminant Analysis"
      pros: "Simple, interpretable"
      cons: "Assumes normal distributions (violated in credit data)"
      decision: "Rejected - Poor statistical fit"

# ============================================================================
# SECTION 2: MATHEMATICAL FORMULATION
# ============================================================================

mathematical_formulation:
  model_equation: |
    Let:
    - y ∈ {0,1} be default indicator (1 = default, 0 = no default)
    - x₁ = log(income)
    - x₂ = credit_score (standardized)
    - x₃ = debt_to_income_ratio
    - x₄ = employment_tenure_months
    
    Logistic Regression Model:
    
    P(y=1 | x) = σ(β₀ + β₁x₁ + β₂x₂ + β₃x₃ + β₄x₄)
    
    Where σ(z) = 1 / (1 + e^(-z)) is the logistic function
    
    Equivalently (log-odds form):
    
    log(P(y=1) / P(y=0)) = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + β₄x₄
  
  feature_transformations:
    income_transformation:
      input: "income (raw annual income in USD)"
      transformation: "log_income = ln(max(income, 1))"
      rationale: |
        Logarithmic transformation because:
        1. Diminishing returns: $1K matters more at $30K income than at $300K
        2. Skewness reduction: Income highly right-skewed, log normalizes
        3. Multiplicative effects: 10% income increase has similar impact across income levels
      formula: "x₁ = ln(max(income, 1))"
      domain: "(0, ∞) → (-∞, ∞)"
    
    credit_score_transformation:
      input: "credit_score (FICO 300-850)"
      transformation: "Standardization (mean-centering)"
      rationale: "Scale to mean=0, std=1 for numerical stability in optimization"
      formula: |
        μ = 680 (mean credit score in training data)
        σ = 100 (standard deviation)
        x₂ = (credit_score - μ) / σ
      domain: "[300, 850] → [-3.8, 1.7]"
    
    dti_transformation:
      input: "debt_to_income_ratio (0-1.5)"
      transformation: "Clipping at 65%"
      rationale: "Prevent extreme leverage ratios from dominating model"
      formula: "x₃ = min(debt_to_income_ratio, 0.65)"
      domain: "[0, 1.5] → [0, 0.65]"
    
    tenure_transformation:
      input: "employment_tenure_months (0-600)"
      transformation: "No transformation (use raw)"
      rationale: "Linear relationship expected; longer tenure → more stable income"
      formula: "x₄ = employment_tenure_months"
      domain: "[0, 600] → [0, 600]"
  
  coefficient_interpretation:
    β₀_intercept:
      expected_value: 1.2
      interpretation: "Baseline log-odds of default when all features = 0"
      economic_meaning: "Reference risk level for standardized borrower"
    
    β₁_log_income:
      expected_value: -0.35
      expected_sign: "negative (MUST be negative for conceptual soundness)"
      interpretation: "1% increase in income → 0.35% decrease in log-odds of default"
      economic_meaning: "Higher income → Better debt service capacity → Lower default risk"
      property_check: "Monotonicity: P(default) decreases with income"
    
    β₂_credit_score:
      expected_value: -0.55
      expected_sign: "negative (MUST be negative)"
      interpretation: "1 std dev increase in credit score → 0.55 decrease in log-odds"
      economic_meaning: "Higher credit score → Better payment history → Lower default risk"
      property_check: "Monotonicity: P(default) decreases with credit score"
    
    β₃_dti:
      expected_value: 0.42
      expected_sign: "positive (MUST be positive)"
      interpretation: "1% increase in DTI → 0.42 increase in log-odds of default"
      economic_meaning: "Higher leverage → Less capacity for new debt → Higher default risk"
      property_check: "Monotonicity: P(default) increases with DTI"
    
    β₄_tenure:
      expected_value: -0.008
      expected_sign: "negative"
      interpretation: "1 month longer tenure → 0.008 decrease in log-odds"
      economic_meaning: "Longer employment → More stable income → Lower default risk"
      property_check: "Monotonicity: P(default) decreases with tenure"
  
  coefficient_estimation:
    method: "Maximum Likelihood Estimation (MLE)"
    optimization_algorithm: "L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)"
    regularization: "L2 (Ridge) with λ=0.01"
    
    regularization_rationale: |
      Prevent overfitting on training data (50K samples).
      L2 penalty shrinks coefficients toward zero, improving generalization.
      λ=0.01 chosen via cross-validation (5-fold CV).
    
    convergence_criteria: "Gradient norm <1e-6"
    max_iterations: 1000

# ============================================================================
# SECTION 3: PROPERTY SPECIFICATIONS (FOR QUVALIDATE)
# ============================================================================

properties:
  # ========== CRITICAL PROPERTIES ==========
  
  - property_id: "PROP-001"
    name: "monotonicity_income"
    property_type: "invariant"
    severity: "critical"
    
    formal_specification: |
      ∀ applicants a, b:
        income(a) > income(b) ∧ 
        credit_score(a) = credit_score(b) ∧
        dti(a) = dti(b) ∧
        tenure(a) = tenure(b)
        ⟹ P_default(a) ≤ P_default(b)
    
    natural_language: "Higher income should monotonically decrease default probability, holding all else equal"
    
    test_strategy: |
      Generate 100 pairs of applicants:
      - Vary only income: applicant_a.income > applicant_b.income
      - Keep constant: credit_score, dti, tenure
      - Assert: predict(a) <= predict(b)
    
    hypothesis_test_template: |
      @given(
          income_low=st.floats(min_value=10000, max_value=200000),
          income_high=st.floats(min_value=10000, max_value=200000),
          credit_score=st.integers(min_value=300, max_value=850),
          dti=st.floats(min_value=0, max_value=0.65),
          tenure=st.integers(min_value=0, max_value=240)
      )
      def test_monotonicity_income(income_low, income_high, credit_score, dti, tenure):
          assume(income_high > income_low)  # Only test when income_high > income_low
          
          prob_high_income = model.predict(income_high, credit_score, dti, tenure)
          prob_low_income = model.predict(income_low, credit_score, dti, tenure)
          
          assert prob_high_income <= prob_low_income, \
              f"Monotonicity violated: income ${income_high} has higher default prob ({prob_high_income:.3f}) than ${income_low} ({prob_low_income:.3f})"
    
    regulatory_basis: "SR 11-7 Section 1: Conceptual Soundness"
    
    failure_implications: |
      If this property fails:
      - Model violates economic theory (nonsensical)
      - SR 11-7 validation would fail (conceptually unsound)
      - Examiner would reject model
      - CRITICAL: Must fix before production
  
  - property_id: "PROP-002"
    name: "monotonicity_credit_score"
    property_type: "invariant"
    severity: "critical"
    
    formal_specification: |
      ∀ a, b: credit_score(a) > credit_score(b) ∧ other_vars_equal ⟹ P_default(a) ≤ P_default(b)
    
    natural_language: "Higher credit score should monotonically decrease default probability"
    
    test_strategy: "Generate 100 pairs differing only in credit_score"
    
    hypothesis_test_template: |
      @given(
          score_low=st.integers(min_value=300, max_value=850),
          score_high=st.integers(min_value=300, max_value=850),
          income=st.floats(min_value=10000, max_value=300000),
          dti=st.floats(min_value=0, max_value=0.65),
          tenure=st.integers(min_value=0, max_value=240)
      )
      def test_monotonicity_credit_score(score_low, score_high, income, dti, tenure):
          assume(score_high > score_low)
          
          prob_high_score = model.predict(income, score_high, dti, tenure)
          prob_low_score = model.predict(income, score_low, dti, tenure)
          
          assert prob_high_score <= prob_low_score, \
              f"Monotonicity violated: score {score_high} → prob {prob_high_score:.3f}, score {score_low} → prob {prob_low_score:.3f}"
    
    regulatory_basis: "SR 11-7 Conceptual Soundness"
  
  - property_id: "PROP-003"
    name: "probability_bounds"
    property_type: "postcondition"
    severity: "critical"
    
    formal_specification: "∀ x ∈ valid_inputs: 0 ≤ P_default(x) ≤ 1"
    
    natural_language: "Model output must be a valid probability"
    
    test_strategy: "Generate 500 random valid inputs, verify output ∈ [0,1]"
    
    hypothesis_test_template: |
      @given(
          income=st.floats(min_value=10000, max_value=500000),
          credit_score=st.integers(min_value=300, max_value=850),
          dti=st.floats(min_value=0, max_value=0.65),
          tenure=st.integers(min_value=0, max_value=240)
      )
      def test_probability_bounds(income, credit_score, dti, tenure):
          prob = model.predict(income, credit_score, dti, tenure)
          
          assert 0 <= prob <= 1, \
              f"Invalid probability: {prob} (must be in [0, 1])"
    
    regulatory_basis: "SR 11-7 Conceptual Soundness"
  
  - property_id: "PROP-004"
    name: "no_prohibited_factors_ecoa"
    property_type: "precondition"
    severity: "critical"
    
    formal_specification: |
      ∀ input x: x ∉ {race, color, religion, national_origin, sex, marital_status, age*}
      *age: except for age of majority verification
    
    natural_language: "Model cannot use ECOA prohibited factors"
    
    test_strategy: |
      Static code analysis:
      1. Parse model source code (AST)
      2. Check for prohibited variable names
      3. Check for proxy variables (zip_code, first_name)
      4. Verify no data leakage through correlated features
    
    implementation: |
      def test_no_prohibited_factors():
          import ast
          import inspect
          
          # Get model source code
          model_source = inspect.getsource(model.predict)
          tree = ast.parse(model_source)
          
          prohibited = ['race', 'ethnicity', 'religion', 'national_origin', 
                       'sex', 'gender', 'marital_status', 'age', 
                       'zip_code', 'zipcode', 'postal_code']
          
          # Check variable names in AST
          for node in ast.walk(tree):
              if isinstance(node, ast.Name):
                  assert node.id.lower() not in prohibited, \
                      f"ECOA VIOLATION: Prohibited factor '{node.id}' found in code"
    
    regulatory_basis: "ECOA 15 U.S.C. § 1691(a)"
    
    failure_implications: |
      CRITICAL REGULATORY VIOLATION
      - Potential ECOA lawsuit
      - Regulatory enforcement action
      - Consent order possible
      - CANNOT DEPLOY TO PRODUCTION
  
  # ========== HIGH PRIORITY PROPERTIES ==========
  
  - property_id: "PROP-005"
    name: "input_validation_comprehensive"
    property_type: "precondition"
    severity: "high"
    
    formal_specification: |
      ∀ input x:
        income > 0 ∧
        300 ≤ credit_score ≤ 850 ∧
        0 ≤ dti ≤ 1.5 ∧
        tenure ≥ 0
    
    natural_language: "All inputs must be within valid business ranges"
    
    test_strategy: "Generate invalid inputs (negative, out-of-range, null), verify rejection with clear error messages"
    
    hypothesis_test_template: |
      def test_rejects_invalid_income():
          with pytest.raises(ValueError, match="income must be positive"):
              model.predict(income=-1000, credit_score=700, dti=0.3, tenure=24)
      
      def test_rejects_invalid_credit_score():
          with pytest.raises(ValueError, match="credit_score must be 300-850"):
              model.predict(income=50000, credit_score=999, dti=0.3, tenure=24)
      
      @given(st.floats(min_value=-100000, max_value=0))
      def test_rejects_all_negative_incomes(bad_income):
          with pytest.raises(ValueError):
              model.predict(income=bad_income, credit_score=700, dti=0.3, tenure=24)
    
    regulatory_basis: "SR 11-7: Data Quality and Integrity"
  
  - property_id: "PROP-006"
    name: "disparate_impact_80_percent_rule"
    property_type: "statistical"
    severity: "critical"
    
    formal_specification: |
      Let:
      - approval_rate(protected) = % of protected class approved
      - approval_rate(control) = % of control group approved
      
      Then: approval_rate(protected) / approval_rate(control) ≥ 0.80
    
    natural_language: "Model must not have disparate impact (80% rule for fair lending)"
    
    test_strategy: |
      1. Generate matched pairs differing only in protected class proxy
      2. Calculate approval rates for each group
      3. Verify ratio ≥80%
      
      Note: Since model cannot use protected class directly (ECOA), we test
      using proxy analysis: do income-matched groups have similar approval rates?
    
    hypothesis_test_template: |
      def test_disparate_impact(test_dataset_with_demographics):
          """
          Test with historical data that includes demographic information
          (used for testing only, NOT as model input)
          """
          
          # Calculate approval rates by group
          results = {}
          for group in ['protected_class', 'control_group']:
              group_data = test_dataset_with_demographics[test_dataset_with_demographics.group == group]
              
              predictions = [
                  model.predict(row.income, row.credit_score, row.dti, row.tenure)
                  for _, row in group_data.iterrows()
              ]
              
              approvals = sum(1 for p in predictions if p < 0.15)  # <15% default prob = approve
              approval_rate = approvals / len(predictions)
              results[group] = approval_rate
          
          # 80% rule
          ratio = results['protected_class'] / results['control_group']
          
          assert ratio >= 0.80, \
              f"Disparate impact detected: {ratio:.2%} (must be ≥80%)"
    
    regulatory_basis: "ECOA Disparate Impact Standard (Regulation B)"
  
  - property_id: "PROP-007"
    name: "robustness_to_small_perturbations"
    property_type: "statistical"
    severity: "high"
    
    formal_specification: |
      ∀ x ∈ valid_inputs, ∀ ε ∈ [-0.01, 0.01]:
        |P_default(x + ε) - P_default(x)| ≤ 0.05
    
    natural_language: "Small input changes (±1%) should not drastically change predictions (>5%)"
    
    test_strategy: "Add ±1% noise to each input, verify output changes <5%"
    
    hypothesis_test_template: |
      @given(
          income=st.floats(min_value=20000, max_value=200000),
          credit_score=st.integers(min_value=400, max_value=800),
          dti=st.floats(min_value=0.1, max_value=0.6),
          tenure=st.integers(min_value=6, max_value=120),
          noise_pct=st.floats(min_value=-0.01, max_value=0.01)
      )
      def test_robustness(income, credit_score, dti, tenure, noise_pct):
          # Original prediction
          prob_original = model.predict(income, credit_score, dti, tenure)
          
          # Perturbed prediction (add noise to income)
          income_noisy = income * (1 + noise_pct)
          prob_noisy = model.predict(income_noisy, credit_score, dti, tenure)
          
          # Verify stability
          change = abs(prob_noisy - prob_original)
          
          assert change <= 0.05, \
              f"Model unstable: {noise_pct:.1%} income change → {change:.1%} probability change"
    
    regulatory_basis: "SR 11-7: Model Stability and Reliability"
  
  # ========== MEDIUM PRIORITY PROPERTIES ==========
  
  - property_id: "PROP-008"
    name: "coefficient_signs_economically_valid"
    property_type: "invariant"
    severity: "medium"
    
    formal_specification: |
      β₁ < 0 ∧  # income coefficient negative
      β₂ < 0 ∧  # credit score coefficient negative
      β₃ > 0 ∧  # DTI coefficient positive
      β₄ < 0    # tenure coefficient negative
    
    natural_language: "Coefficient signs must match economic theory"
    
    test_strategy: "After model training, inspect coefficients, verify signs"
    
    implementation: |
      def test_coefficient_signs():
          coeffs = model.get_coefficients()
          
          assert coeffs['log_income'] < 0, "Income coeff must be negative"
          assert coeffs['credit_score'] < 0, "Credit score coeff must be negative"
          assert coeffs['dti'] > 0, "DTI coeff must be positive"
          assert coeffs['tenure'] < 0, "Tenure coeff must be negative"
    
    regulatory_basis: "SR 11-7: Conceptual Soundness (economic intuition)"
  
  - property_id: "PROP-009"
    name: "adverse_action_completeness"
    property_type: "postcondition"
    severity: "critical"
    
    formal_specification: |
      ∀ x: decision(x) = "DENIED" ⟹ adverse_action_reasons(x) ≠ ∅
    
    natural_language: "If application denied, must provide specific reasons (ECOA requirement)"
    
    test_strategy: "Generate denied applications, verify adverse action reasons provided"
    
    hypothesis_test_template: |
      @given(
          income=st.floats(min_value=10000, max_value=40000),  # Low income → likely denied
          credit_score=st.integers(min_value=300, max_value=600),  # Low score → likely denied
          dti=st.floats(min_value=0.5, max_value=0.8),  # High DTI → likely denied
          tenure=st.integers(min_value=0, max_value=12)  # Short tenure → likely denied
      )
      def test_adverse_action_reasons(income, credit_score, dti, tenure):
          result = model.predict_with_decision(income, credit_score, dti, tenure)
          
          if result['decision'] == 'DENIED':
              assert result['adverse_action_reasons'] is not None, \
                  "ECOA VIOLATION: Denied application missing adverse action reasons"
              
              assert len(result['adverse_action_reasons']) >= 1, \
                  "Must provide at least one reason for denial"
              
              # Reasons must be specific
              assert all(len(reason) > 10 for reason in result['adverse_action_reasons']), \
                  "Adverse action reasons must be specific, not generic"
    
    regulatory_basis: "ECOA § 1691(d): Adverse Action Notice Requirement"

# ============================================================================
# SECTION 4: VALIDATION METHODOLOGY
# ============================================================================

validation_methodology:
  back_testing:
    approach: "Out-of-time validation"
    training_period: "2020-01-01 to 2024-06-30"
    test_period: "2024-07-01 to 2024-12-31"
    
    performance_metrics:
      - metric: "AUC-ROC"
        target: ">0.75"
        interpretation: "Model discriminates defaulters from non-defaulters"
      
      - metric: "KS Statistic (Kolmogorov-Smirnov)"
        target: ">0.30"
        interpretation: "Maximum separation between default/non-default distributions"
      
      - metric: "Gini Coefficient"
        target: ">0.50"
        interpretation: "Model rank-ordering power (2*AUC - 1)"
      
      - metric: "Calibration (Hosmer-Lemeshow test)"
        target: "p-value >0.05"
        interpretation: "Predicted probabilities match observed default rates"
    
    peer_benchmarking:
      - "Compare against bureau generic score (FICO)"
      - "Compare against prior generation internal model"
      - "Target: New model AUC ≥ Bureau score AUC"
  
  stress_testing:
    scenarios:
      - scenario: "Economic Recession"
        assumptions: "Unemployment +5%, income -10%, defaults +50%"
        test: "Model predictions increase appropriately"
      
      - scenario: "Credit Bureau Data Error"
        assumptions: "Credit scores incorrect ±50 points"
        test: "Model gracefully handles (flags for review)"
      
      - scenario: "Extreme Outliers"
        assumptions: "Income $10M, DTI 300%"
        test: "Model doesn't crash, returns bounded probability"
  
  sensitivity_analysis:
    - parameter: "β₁ (income coefficient)"
      variation: "±20%"
      impact_on: "Default rate predictions"
      acceptable_range: "Approval rate changes <10%"

# ============================================================================
# SECTION 5: MONITORING & MAINTENANCE
# ============================================================================

monitoring_specification:
  ongoing_validation:
    frequency: "Quarterly"
    
    metrics_tracked:
      - "Default rate: Predicted vs Actual (within ±2%)"
      - "Approval rate: Overall and by segment"
      - "Adverse action rate: % of denials (track trends)"
      - "Model stability: PSI (Population Stability Index) <0.15"
    
    alert_thresholds:
      - condition: "Actual default rate >120% of predicted"
        action: "Flag for MRM review"
      
      - condition: "PSI >0.25"
        action: "Immediate model revalidation required"
      
      - condition: "Disparate impact ratio <0.75"
        action: "URGENT: Potential ECOA violation, halt usage"
  
  model_refresh_triggers:
    - "PSI >0.25 (population shift)"
    - "Performance degradation >10% (AUC drops)"
    - "Regulatory change (new ECOA guidance)"
    - "Business change (new loan products)"
    
  documentation_updates:
    frequency: "With each model refresh or annually (whichever is sooner)"
    requirements:
      - "Update back-test results with latest data"
      - "Update coefficient values if retrained"
      - "Document any changes to validation approach"
      - "Obtain MRM re-approval"

# ============================================================================
# SECTION 6: IMPLEMENTATION GUIDANCE
# ============================================================================

implementation_guidance:
  model_training_steps:
    step_1:
      task: "Data Preparation"
      details: |
        1. Extract loan data from warehouse (2020-2024)
        2. Define default: 90+ days delinquent within 12 months
        3. Clean data: Remove nulls, outliers
        4. Feature engineering: log(income), standardize credit_score
        5. Train/test split: 80/20 (stratified by default rate)
    
    step_2:
      task: "Model Training"
      details: |
        1. Use scikit-learn LogisticRegression
        2. Regularization: L2 with λ=0.01
        3. Solver: 'lbfgs'
        4. Max iterations: 1000
        5. Class weights: Balanced (account for 17% default rate imbalance)
    
    step_3:
      task: "Coefficient Validation"
      details: |
        1. Inspect coefficient signs (match economic theory)
        2. Check coefficient magnitudes (reasonable ranges)
        3. Statistical significance: p-values <0.05 for all coefficients
        4. VIF (Variance Inflation Factor) <5 (no multicollinearity)
    
    step_4:
      task: "Performance Validation"
      details: |
        1. Calculate AUC-ROC on test set (target >0.75)
        2. Calibration plot: Predicted vs actual default rates
        3. Confusion matrix at decision threshold (15% default prob)
        4. Feature importance: SHAP values for explainability
    
    step_5:
      task: "Property-Based Testing (QuValidate)"
      details: |
        1. Run all critical properties (PROP-001 to PROP-009)
        2. Generate 100+ test cases per property
        3. Verify all properties pass
        4. Document any failures, iterate until pass
    
    step_6:
      task: "Documentation Generation"
      details: |
        1. SR 11-7 model documentation (auto-generated by QCreate)
        2. ECOA compliance memo (fair lending analysis)
        3. Back-test results report
        4. Property test results (QuValidate report)
  
  code_example:
    language: "Python"
    framework: "scikit-learn"
    template: |
      import numpy as np
      import pandas as pd
      from sklearn.linear_model import LogisticRegression
      from sklearn.preprocessing import StandardScaler
      import shap
      
      class ConsumerCreditScoringModel:
          """
          Consumer credit scoring model using logistic regression.
          
          Theoretical Basis:
          - Merton structural model: Default when assets < liabilities
          - Proxies: Income (asset stream), DTI (liability level)
          
          Key Assumptions:
          - Historical default patterns (2020-2024) predict future
          - Credit bureau data accurate (FCRA compliance)
          - Macroeconomic conditions stable
          
          Limitations:
          - Trained on pre-pandemic data (may not capture new patterns)
          - Linear relationships (no feature interactions)
          - Consumer loans only ($5K-$50K range)
          
          Regulatory:
          - SR 11-7 compliant (conceptual soundness, validation, monitoring)
          - ECOA compliant (no prohibited factors, adverse actions provided)
          """
          
          def __init__(self):
              self.model = LogisticRegression(
                  penalty='l2',
                  C=1/0.01,  # Inverse of lambda=0.01
                  solver='lbfgs',
                  max_iter=1000,
                  class_weight='balanced'
              )
              self.scaler = StandardScaler()
              self.explainer = None  # SHAP explainer (for adverse actions)
          
          def train(self, X_train, y_train):
              """Train model on historical data"""
              
              # Feature engineering
              X_transformed = self._transform_features(X_train)
              
              # Fit model
              self.model.fit(X_transformed, y_train)
              
              # Initialize SHAP explainer
              self.explainer = shap.LinearExplainer(self.model, X_transformed)
              
              return self
          
          def predict(self, income: float, credit_score: int, 
                     dti: float, tenure: int) -> float:
              """
              Predict default probability
              
              Args:
                  income: Annual gross income (USD)
                  credit_score: FICO score (300-850)
                  dti: Debt-to-income ratio (0-1.5)
                  tenure: Employment tenure (months)
              
              Returns:
                  Default probability [0, 1]
              
              Raises:
                  ValueError: If inputs invalid
              """
              
              # Input validation (PROP-005)
              self._validate_inputs(income, credit_score, dti, tenure)
              
              # Transform features
              X = self._transform_features(
                  pd.DataFrame([{
                      'income': income,
                      'credit_score': credit_score,
                      'dti': dti,
                      'tenure': tenure
                  }])
              )
              
              # Predict probability
              prob = self.model.predict_proba(X)[0, 1]
              
              # Ensure bounds (PROP-003)
              assert 0 <= prob <= 1, f"Invalid probability: {prob}"
              
              return prob
          
          def predict_with_decision(self, income, credit_score, dti, tenure):
              """
              Predict with credit decision and adverse action reasons
              
              Required for ECOA compliance
              """
              
              prob = self.predict(income, credit_score, dti, tenure)
              
              # Decision thresholds
              if prob < 0.10:
                  decision = "APPROVED"
                  reasons = None
              elif prob < 0.20:
                  decision = "REVIEW_REQUIRED"
                  reasons = None
              else:
                  decision = "DENIED"
                  reasons = self._generate_adverse_action_reasons(
                      income, credit_score, dti, tenure, prob
                  )
              
              return {
                  'default_probability': prob,
                  'decision': decision,
                  'adverse_action_reasons': reasons
              }
          
          def _validate_inputs(self, income, credit_score, dti, tenure):
              """Validate all inputs (PROP-005)"""
              
              if income <= 0:
                  raise ValueError(f"income must be positive, got {income}")
              
              if not (300 <= credit_score <= 850):
                  raise ValueError(f"credit_score must be 300-850, got {credit_score}")
              
              if not (0 <= dti <= 1.5):
                  raise ValueError(f"dti must be 0-1.5, got {dti}")
              
              if tenure < 0:
                  raise ValueError(f"tenure must be non-negative, got {tenure}")
          
          def _transform_features(self, X: pd.DataFrame) -> np.ndarray:
              """Apply feature transformations"""
              
              X_transformed = X.copy()
              
              # Log income
              X_transformed['log_income'] = np.log(np.maximum(X['income'], 1))
              
              # Standardize credit score
              X_transformed['credit_score_scaled'] = (X['credit_score'] - 680) / 100
              
              # Clip DTI
              X_transformed['dti_clipped'] = np.minimum(X['dti'], 0.65)
              
              # Tenure (no transform)
              X_transformed['tenure'] = X['tenure']
              
              return X_transformed[['log_income', 'credit_score_scaled', 
                                   'dti_clipped', 'tenure']].values
          
          def _generate_adverse_action_reasons(self, income, credit_score, 
                                               dti, tenure, prob):
              """
              Generate specific adverse action reasons (ECOA requirement)
              
              Uses SHAP to identify which features most contributed to denial
              """
              
              # Get SHAP values
              X = self._transform_features(pd.DataFrame([{
                  'income': income,
                  'credit_score': credit_score,
                  'dti': dti,
                  'tenure': tenure
              }]))
              
              shap_values = self.explainer.shap_values(X)[0]
              
              # Top negative contributors (pushed toward denial)
              feature_contributions = {
                  'income': shap_values[0],
                  'credit_score': shap_values[1],
                  'dti': shap_values[2],
                  'tenure': shap_values[3]
              }
              
              # Sort by absolute contribution (most impactful)
              sorted_features = sorted(
                  feature_contributions.items(),
                  key=lambda x: abs(x[1]),
                  reverse=True
              )
              
              # Generate reasons
              reasons = []
              for feature, contribution in sorted_features[:3]:  # Top 3
                  if feature == 'credit_score' and credit_score < 640:
                      reasons.append(f"Credit score {credit_score} below recommended minimum (640)")
                  elif feature == 'income' and income < 35000:
                      reasons.append(f"Income ${income:,.0f} insufficient for loan amount")
                  elif feature == 'dti' and dti > 0.43:
                      reasons.append(f"Debt-to-income ratio {dti:.1%} exceeds guideline (43%)")
                  elif feature == 'tenure' and tenure < 12:
                      reasons.append(f"Employment history {tenure} months below minimum (12)")
              
              return reasons if reasons else ["Overall credit profile indicates high default risk"]

# ============================================================================
# SECTION 7: QUVALIDATE INTEGRATION
# ============================================================================

quvalidate_configuration:
  property_tests_to_run:
    - "PROP-001: monotonicity_income (100 test cases)"
    - "PROP-002: monotonicity_credit_score (100 test cases)"
    - "PROP-003: probability_bounds (500 test cases)"
    - "PROP-004: no_prohibited_factors_ecoa (static analysis)"
    - "PROP-005: input_validation_comprehensive (50 test cases)"
    - "PROP-006: disparate_impact_80_percent_rule (requires test dataset)"
    - "PROP-007: robustness_to_small_perturbations (100 test cases)"
    - "PROP-008: coefficient_signs_economically_valid (post-training check)"
    - "PROP-009: adverse_action_completeness (100 test cases)"
  
  expected_validation_time: "3-5 minutes (parallel execution)"
  
  matlab_benchmark:
    enabled: true
    reference_implementation: "credit_scoring_reference.m"
    tolerance: 1.0e-6
    comparison_metrics:
      - "Numerical accuracy (match within tolerance)"
      - "Performance (Python speedup vs MATLAB)"
      - "Memory usage (Python vs MATLAB)"
  
  success_criteria:
    - "All critical properties (severity: critical) must pass"
    - "≥90% of high priority properties must pass"
    - "Overall QuValidate score ≥90/100"
    - "Zero ECOA violations detected"
    - "MATLAB numerical match (if reference exists)"

# ============================================================================
# SECTION 8: DEPLOYMENT CHECKLIST
# ============================================================================

deployment_checklist:
  pre_deployment:
    - task: "QuValidate validation complete"
      status: "required"
      owner: "Developer"
    
    - task: "MRM approval obtained"
      status: "required"
      owner: "Model Risk Management"
    
    - task: "Integration testing (LOS system)"
      status: "required"
      owner: "IT / DevOps"
    
    - task: "User acceptance testing (underwriters)"
      status: "required"
      owner: "Business"
    
    - task: "Monitoring dashboard configured"
      status: "required"
      owner: "DevOps / Risk"
  
  post_deployment:
    - task: "Shadow mode (parallel with existing model, 2 weeks)"
      validation: "Compare decisions, ensure consistency"
    
    - task: "Phased rollout (10% → 50% → 100% traffic)"
      timeline: "4 weeks"
    
    - task: "First quarterly validation (3 months post-deployment)"
      deliverable: "Outcomes analysis report for MRM"
