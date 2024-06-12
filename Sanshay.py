using System;
using System.Collections.Generic;
using System.Linq;

namespace AdvancedRiskManagementProgram
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Welcome to the Advanced Risk Management Program!");

            var userInput = new Dictionary<string, string>
            {
                {"name", Console.ReadLine()},
                {"age", Console.ReadLine()},
                {"occupation", Console.ReadLine()},
                {"income", Console.ReadLine()},
                {"creditScore", Console.ReadLine()},
                {"investmentExperience", Console.ReadLine()},
                {"riskTolerance", Console.ReadLine()},
                {"insuranceType", Console.ReadLine()},
                {"coverageAmount", Console.ReadLine()},
                {"policyDuration", Console.ReadLine()},
                {"dependents", Console.ReadLine()},
                {"assets", Console.ReadLine()},
                {"liabilities", Console.ReadLine()}
            };

            var premium = userInput["insuranceType"] switch
            {
                "Life" => CalculateLifePremium(userInput),
                "Health" => CalculateHealthPremium(userInput),
                "Auto" => CalculateAutoPremium(userInput),
                "Home" => CalculateHomePremium(userInput),
                _ => throw new ArgumentException("Invalid insurance type")
            };

            var insurancePlan = new InsurancePlan
            {
                UserName = userInput["name"],
                InsuranceType = userInput["insuranceType"],
                CoverageAmount = decimal.Parse(userInput["coverageAmount"]),
                PolicyDuration = int.Parse(userInput["policyDuration"]),
                Premium = premium
            };

            Console.WriteLine("Insurance Plan:");
            Console.WriteLine($"User Name: {insurancePlan.UserName}");
            Console.WriteLine($"Insurance Type: {insurancePlan.InsuranceType}");
            Console.WriteLine($"Coverage Amount: {insurancePlan.CoverageAmount:C}");
            Console.WriteLine($"Policy Duration: {insurancePlan.PolicyDuration} years");
            Console.WriteLine($"Premium: {insurancePlan.Premium:C}");
        }

        static decimal CalculateLifePremium(Dictionary<string, string> userInput)
        {
            var age = int.Parse(userInput["age"]);
            var occupation = userInput["occupation"];
            var income = decimal.Parse(userInput["income"]);
            var creditScore = int.Parse(userInput["creditScore"]);
            var investmentExperience = userInput["investmentExperience"];
            var riskTolerance = userInput["riskTolerance"];
            var coverageAmount = decimal.Parse(userInput["coverageAmount"]);
            var policyDuration = int.Parse(userInput["policyDuration"]);
            var dependents = int.Parse(userInput["dependents"]);
            var assets = decimal.Parse(userInput["assets"]);
            var liabilities = decimal.Parse(userInput["liabilities"]);

            return (age * 0.05m) + (occupation == "High-Risk" ? 0.1m : 0.05m) + (income * 0.01m) + (creditScore * 0.001m) + (investmentExperience == "Advanced" ? 0.05m : 0.01m) + (riskTolerance == "Aggressive" ? 0.1m : 0.05m) + (coverageAmount * 0.01m) + (policyDuration * 0.05m) + (dependents * 0.01m) + (assets * 0.005m) - (liabilities * 0.005m);
        }

        static decimal CalculateHealthPremium(Dictionary<string, string> userInput)
        {
            var age = int.Parse(userInput["age"]);
            var occupation = userInput["occupation"];
            var income = decimal.Parse(userInput["income"]);
            var creditScore = int.Parse(userInput["creditScore"]);
            var investmentExperience = userInput["investmentExperience"];
            var riskTolerance = userInput["riskTolerance"];
            var coverageAmount = decimal.Parse(userInput["coverageAmount"]);
            var policyDuration = int.Parse(userInput["policyDuration"]);
            var dependents = int.Parse(userInput["dependents"]);
            var assets = decimal.Parse(userInput["assets"]);
            var liabilities = decimal.Parse(userInput["liabilities"]);

            return (age * 0.03m) + (occupation == "High-Risk" ? 0.05m : 0.02m) + (income * 0.005m) + (creditScore * 0.0005m) + (investmentExperience == "Advanced" ? 0.02m : 0.005m) + (riskTolerance == "Aggressive" ? 0.05m : 0.02m) + (coverageAmount * 0.005m) + (policyDuration * 0.02m) + (dependents * 0.005m) + (assets * 0.0025m) - (liabilities * 0.0025m);
        }

        static decimal CalculateAutoPremium(Dictionary<string, string> userInput)
        {
            var age = int.Parse(userInput["age"]);
            var occupation = userInput["occupation"];
            var income = decimal.Parse(userInput["income"]);
            var creditScore = int.Parse(userInput["creditScore"]);
            var investmentExperience = userInput["investmentExperience"];
            var riskTolerance = userInput["riskTolerance"];
            var coverageAmount = decimal.Parse(userInput["coverageAmount"]);
            var policyDuration = int.Parse(userInput["policyDuration"]);
            var dependents = int.Parse(userInput["dependents"]);
            var assets = decimal.Parse(userInput["assets"]);
            var liabilities = decimal.Parse(userInput["liabilities"]);

            return (age * 0.02m) + (occupation == "High-Risk"? 0.03m : 0.01m) + (income * 0.002m) + (creditScore * 0.0002m) + (investmentExperience == "Advanced"? 0.01m : 0.002m) + (riskTolerance == "Aggressive"? 0.03m : 0.01m) + (coverageAmount * 0.002m) + (policyDuration * 0.01m) + (dependents * 0.002m) + (assets * 0.00125m) - (liabilities * 0.00125m);
        }

        static decimal CalculateHomePremium(Dictionary<string, string> userInput)
        {
            var age = int.Parse(userInput["age"]);
            var occupation = userInput["occupation"];
            var income = decimal.Parse(userInput["income"]);
            var creditScore = int.Parse(userInput["creditScore"]);
            var investmentExperience = userInput["investmentExperience"];
            var riskTolerance = userInput["riskTolerance"];
            var coverageAmount = decimal.Parse(userInput["coverageAmount"]);
            var policyDuration = int.Parse(userInput["policyDuration"]);
            var dependents = int.Parse(userInput["dependents"]);
            var assets = decimal.Parse(userInput["assets"]);
            var liabilities = decimal.Parse(userInput["liabilities"]);

            return (age * 0.01m) + (occupation == "High-Risk"? 0.02m : 0.005m) + (income * 0.001m) + (creditScore * 0.0001m) +(investmentExperience == "Advanced"? 0.005m : 0.001m) + (riskTolerance == "Aggressive"? 0.02m : 0.005m) + (coverageAmount * 0.001m) + (policyDuration * 0.005m) + (dependents * 0.001m) + (assets * 0.000625m) - (liabilities * 0.000625m);
        }
    }

    public class InsurancePlan
    {
        public string UserName { get; set; }
        public string InsuranceType { get; set; }
        public decimal CoverageAmount { get; set; }
        public int PolicyDuration { get; set; }
        public decimal Premium { get; set; }
    }
}
