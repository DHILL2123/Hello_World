# **Beyond the Fairway: Predicting the Future of Driving Distance in Golf ⛳**  
**Author:** Deadrien Hill  
**Project:** D214 Task 3 – Time Series Modeling  

## **📌 Project Overview**
This project evaluates the **predictability of driving distance (Drive Avg)** in professional golf using **Holt’s Exponential Smoothing Model**. The objective was to determine whether **driving distance can be forecasted with at least 90% accuracy** and to assess its impact on **player performance, regulations, and competitive balance**.

## **🎯 Hypothesis**
- **Null Hypothesis (H₀):** The average driving distance **cannot** be predicted with 90% accuracy.  
- **Alternative Hypothesis (H₁):** The average driving distance **can** be predicted with an accuracy significantly greater than 90%.

## **📊 Data Collection & Preprocessing**
- **Dataset:** Historical **PGA Tour statistics (1987–2025)**, obtained via **web scraping and manual downloads**.
- **Cleaning Steps:**
  - Merged datasets by **player-year identifiers**.
  - Addressed **missing values** using **column mean imputation**.
  - Removed **duplicates** to maintain data integrity.

## **🧠 Analytical Methods & Models**
1. **Descriptive Statistics:**  
   - Calculated **mean (282.15 yards), standard deviation (16.04 yards), min & max values**.  
2. **Correlation Analysis:**  
   - Found a **negative correlation (-0.54) between Drive Avg and Par 5 scoring**, meaning **longer drives improve Par 5 performance**.  
3. **K-Means Clustering:**  
   - Segmented players into **3 clusters** based on **driving distance and performance**:  
     - **Cluster 0:** Elite long hitters  
     - **Cluster 1:** Moderate performers  
     - **Cluster 2:** Shorter hitters with higher scores  
4. **Time Series Forecasting (Holt’s Model):**  
   - Predicted future driving distances with **R² = 0.978** (strong predictive capability).  
5. **Hypothesis Testing:**  
   - **T-statistic:** 2.27  
   - **P-value:** 0.0147 (**statistically significant, rejects H₀**).

## **📈 Key Findings**
- The current **mean driving distance** is **282.15 yards** (SD = 16.04 yards).  
- **Projected driving distances:**
  - **2025:** 302.02 yards  
  - **2026:** 302.75 yards  
  - **2027:** 303.49 yards  
  - **2028:** 304.22 yards  
  - **2029:** 304.95 yards  
- **Regulatory Insight:** Despite continued increases, projected distances remain **below the USGA’s 317-yard limit** through 2029.

## **📊 Visualizations**
✔ **Trend in Driving Distance Over Time:** 📈 (Steady increase, does not exceed 317-yard limit)  
✔ **Correlation Heatmap:** 🔥 (Negative correlation between Drive Avg & Par 5 scoring)  
✔ **Scatterplot:** 📌 (Longer drives improve Par 5 performance)  
✔ **Forecasted Distances (2025-2029):** 📊 (Growth remains within regulation limits)  

## **⚠ Limitations**
- **K-Means Clustering** assumes **spherical clusters**, which may oversimplify performance trends.  
- **Holt’s Model** relies on **past trends**, not factoring in **equipment rule changes, biomechanics advancements, or weather conditions**.  
- **Selection Bias:** Data **excludes amateur golfers**, limiting generalizability beyond PGA professionals.

## **🏌 Proposed Actions & Strategic Benefits**
### **📌 Regulatory Recommendations**
- **USGA should monitor trends** before enforcing equipment restrictions.  
- Consider **course modifications** (e.g., longer layouts, hazard adjustments) to **preserve competitive balance**.

### **📌 Player Development**
- **Coaches** should focus on:
  - **Swing efficiency**
  - **Clubhead speed**
  - **Strategic shot selection**
- **Cluster 2 (short hitters)** must prioritize **distance improvements** to remain competitive.

### **📌 Industry & Business Impact**
- **Golf governing bodies** can use data for **policy decisions**.  
- **Players & coaches** can optimize **training methodologies**.  
- **Equipment manufacturers** can anticipate **regulatory shifts** and adjust **product designs** accordingly.

## **✅ Conclusion**
This study confirms that **driving distance is highly predictable** using **Holt’s Exponential Smoothing Model**, with an **R² of 0.978**. While **driving distances are projected to increase**, they will **not exceed the USGA’s 317-yard limit** within the next five years. However, **future technological advancements** may **necessitate regulatory discussions** to maintain fair competition.  

---

## **📚 References**
- **PGA Tour Statistics (2023)**
- **USGA Distance Insights Report**
- **Machine Learning & Statistical Forecasting Texts**  
  - Brownlee, J. (2018). *Introduction to Time Series Forecasting with Python*.  
  - Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles & Practice (3rd ed.)*.
  - McKinney, W. (2021). *Pandas: Python Data Analysis Library*.

---

### **📬 Contact**
**Author:** Deadrien Hill  
📧 [deadrienhill@yahoo.com]  
🔗 [https://ww.linkedin.com/in/deadrien-hill]
 

---


