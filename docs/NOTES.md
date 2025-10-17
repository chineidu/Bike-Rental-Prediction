# Dynamic Pricing

- Dynamic pricing is a strategy where prices are adjusted in real-time based on various factors to optimize revenue and demand. In the context of bike rental demand forecasting, it combines predictive analytics with pricing strategies to set optimal rental prices.
- The variables shown below play a crucial role in determining the dynamic pricing strategy for bike rentals.

## Table of Variables
<!-- TOC -->

- [Dynamic Pricing](#dynamic-pricing)
  - [Table of Variables](#table-of-variables)
  - [Elasticity](#elasticity)
    - [Elasticity Interpretation](#elasticity-interpretation)
  - [Weather Impact Factors](#weather-impact-factors)
  - [Utilization Rate](#utilization-rate)
  - [Surge](#surge)
  - [Competitor Factor](#competitor-factor)

<!-- /TOC -->

## Elasticity

- This is the responsiveness of demand to changes in price.

- Formula:

$$ Elasticity = \frac{\% \Delta \text{Quantity Demanded}}{\% \Delta \text{Price}} $$

- For example, a 10% increase in rental price might lead to a 2% decrease in bike rentals, indicating an elasticity of -0.2.

$$ Elasticity = \frac{-2\%}{10\%} = -0.2 $$

- In relation to the bike rental data, elasticity can help us understand how changes in factors like temperature, weather conditions, and rental prices impact overall demand.

### Elasticity Interpretation

- Elasticity > 1: Demand is elastic (sensitive to price changes)
- Elasticity < 1: Demand is inelastic (less sensitive to price changes)
- Elasticity = 1: Unit elastic (proportional response to price changes)

## Weather Impact Factors

- Weather conditions have a significant effect on bike rental demand, which can be quantified using impact factors assigned to different weather scenarios.

- Severe weather (heavy rain, snow) reduces rentals more than mild weather conditions.

- Lower impact factors indicate favorable weather that boosts rentals, while higher factors represent adverse conditions that suppress demand.

  - Overall Impact Factors:
    - Good weather (Clear, Few clouds): < 0.8
    - Moderate weather (Scattered clouds, Broken clouds): 0.8 - 1.2
    - Poor weather (Shower rain, Rain, Thunderstorm): > 1.2
    - Severe weather (Snow, Mist): > 1.5

## Utilization Rate

- This is the ratio of actual bike rentals to the total available bikes, indicating how effectively the fleet is being used.

- Formula:

$$ Utilization \, Rate = \frac{\text{Total Rentals}}{\text{Total Bikes}} $$

- A higher utilization rate suggests that the fleet is well-matched to demand, while a lower rate may indicate overcapacity or misalignment with customer needs.
- Ideally prices should be adjusted to maintain an optimal utilization rate, balancing supply and demand. i.e. increasing prices when utilization is high and decreasing them when it's low.

## Surge

- Surge pricing is a dynamic pricing strategy that increases rental prices during periods of high demand to balance supply and demand.

## Competitor Factor

- This factor considers the pricing strategies of competitors in the bike rental market.
- If competitors lower their prices, it may necessitate a price adjustment to remain competitive.
- Conversely, if competitors raise prices, there may be an opportunity to increase prices without losing customers.
