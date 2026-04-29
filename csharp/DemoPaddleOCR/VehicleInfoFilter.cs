using System;
using System.Text.RegularExpressions;

namespace Matsuwa
{
    public class VehicleInfo
    {
        public string Year { get; set; }
        public string Make { get; set; }
        public string Model { get; set; }
        public string Color { get; set; }
        public string Style { get; set; }

        public bool IsEmpty =>
            string.IsNullOrEmpty(Year) &&
            string.IsNullOrEmpty(Make) &&
            string.IsNullOrEmpty(Model) &&
            string.IsNullOrEmpty(Color) &&
            string.IsNullOrEmpty(Style);

        public override string ToString()
        {
            if (IsEmpty) return string.Empty;

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("=== Vehicle Info ===");
            if (!string.IsNullOrEmpty(Year))  sb.AppendLine($"Year:  {Year}");
            if (!string.IsNullOrEmpty(Make))  sb.AppendLine($"Make:  {Make}");
            if (!string.IsNullOrEmpty(Model)) sb.AppendLine($"Model: {Model}");
            if (!string.IsNullOrEmpty(Style)) sb.AppendLine($"Style: {Style}");
            if (!string.IsNullOrEmpty(Color)) sb.AppendLine($"Color: {Color}");
            
            return sb.ToString().TrimEnd();
        }
    }

    public static class VehicleInfoFilter
    {
        private static readonly string[] LineBreaks = new[] { "\r\n", "\n", "\r" };

        public static VehicleInfo Extract(string ocrText)
        {
            var info = new VehicleInfo();
            if (string.IsNullOrWhiteSpace(ocrText)) return info;

            // Check if *** VEHICLE INFO *** marker is present
            if (!ocrText.Contains("VEHICLE INFO", StringComparison.OrdinalIgnoreCase))
                return info;

            // Split into individual lines
            string[] lines = ocrText.Split(LineBreaks, StringSplitOptions.None);

            // Find the index of the marker line
            int startIndex = -1;
            for (int i = 0; i < lines.Length; i++)
            {
                if (lines[i].Contains("VEHICLE INFO", StringComparison.OrdinalIgnoreCase))
                {
                    startIndex = i + 1;
                    break;
                }
            }

            if (startIndex < 0) return info;

            // Rejoin the lines after the marker and fix the "Mode\nl: FIS" → "Model: FIS" split
            // Strategy: iterate lines, try to match known field prefixes, handle broken lines
            string[] fieldLines = ReassembleLines(lines, startIndex);

            foreach (string line in fieldLines)
            {
                string trimmed = line.Trim();
                if (string.IsNullOrEmpty(trimmed)) continue;

                if (TryExtractField(trimmed, "Year", out string year))
                    info.Year = year;
                else if (TryExtractField(trimmed, "Make", out string make))
                    info.Make = ExtendMakerName(make);
                else if (TryExtractField(trimmed, "Model", out string model))
                    info.Model = model;
                else if (TryExtractField(trimmed, "Color", out string color))
                    info.Color = color;
                else if (TryExtractField(trimmed, "Style", out string style))
                    info.Style = style;
            }

            return info;
        }

        /// <summary>
        /// Reassembles lines, merging broken field lines such as:
        ///   "Mode"  followed by  "l: FIS"  → "Model: FIS"
        /// The merge heuristic: if a line looks like the tail of a known field name
        /// (i.e. it does not itself start with a known field prefix and the concatenation
        /// of it with the previous line produces a recognized field pattern), merge them.
        /// </summary>
        private static string[] ReassembleLines(string[] lines, int startIndex)
        {
            // Known field prefixes (full names)
            string[] knownFields = { "Year", "Make", "Model", "Color", "Style" };

            var result = new System.Collections.Generic.List<string>();

            for (int i = startIndex; i < lines.Length; i++)
            {
                string current = lines[i].Trim();
                if (string.IsNullOrEmpty(current)) continue;

                // Try to merge with previous accumulated line if needed
                if (result.Count > 0)
                {
                    string prev = result[result.Count - 1];
                    // If the previous line is not yet a complete "Field: value" line,
                    // try merging: prev + current might form one.
                    if (!IsCompleteFieldLine(prev, knownFields))
                    {
                        // OCR often confuses lowercase 'l' with '1'; normalize the
                        // continuation fragment before attempting the merge so that
                        // e.g. "Mode" + "1: FIS" is treated as "Mode" + "l: FIS" → "Model: FIS".
                        string normalizedCurrent = NormalizeOcrFragment(current);

                        string merged = prev + normalizedCurrent;
                        if (IsCompleteFieldLine(merged, knownFields) || StartsWithKnownField(merged, knownFields))
                        {
                            result[result.Count - 1] = merged;
                            continue;
                        }
                    }
                }

                result.Add(current);
            }

            return result.ToArray();
        }

        /// <summary>
        /// Corrects common single-character OCR misreads at the start of a continuation
        /// fragment that is about to be concatenated with the previous partial line.
        /// For example, "1: FIS" → "l: FIS" so "Mode" + "l: FIS" = "Model: FIS".
        /// </summary>
        private static string NormalizeOcrFragment(string fragment)
        {
            if (string.IsNullOrEmpty(fragment)) return fragment;

            // "1: ..." → "l: ..."  (digit one → lowercase L)
            if (fragment.Length >= 2 && fragment[0] == '1' && fragment[1] == ':')
                return "l" + fragment.Substring(1);

            return fragment;
        }

        private static bool IsCompleteFieldLine(string line, string[] knownFields)
        {
            // A complete line looks like "FieldName: value"
            foreach (string field in knownFields)
            {
                if (line.StartsWith(field + ":", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        private static bool StartsWithKnownField(string line, string[] knownFields)
        {
            foreach (string field in knownFields)
            {
                if (line.StartsWith(field, StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        private static readonly System.Collections.Generic.Dictionary<string, string> MakerAliases =
            new System.Collections.Generic.Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                { "ACUR",  "Acura" },
                { "ALFA",  "Alfa Romeo" },
                { "AUDI",  "Audi" },
                { "BMW",   "BMW" },
                { "BUIC",  "Buick" },
                { "CADI",  "Cadillac" },
                { "CHEV",  "Chevrolet" },
                { "CHRY",  "Chrysler" },
                { "DODG",  "Dodge" },
                { "FIAT",  "Fiat" },
                { "FORD",  "Ford" },
                { "GMC",   "GMC" },
                { "HOND",  "Honda" },
                { "HYUN",  "Hyundai" },
                { "INFI",  "Infiniti" },
                { "JAGU",  "Jaguar" },
                { "JEEP",  "Jeep" },
                { "KIA",   "Kia" },
                { "LAND",  "Land Rover" },
                { "LEXU",  "Lexus" },
                { "LINC",  "Lincoln" },
                { "MAZD",  "Mazda" },
                { "MERC",  "Mercedes-Benz" },
                { "MINI",  "MINI" },
                { "MITS",  "Mitsubishi" },
                { "NISS",  "Nissan" },
                { "PONT",  "Pontiac" },
                { "PORS",  "Porsche" },
                { "RAM",   "Ram" },
                { "SATU",  "Saturn" },
                { "SCION", "Scion" },
                { "SUBA",  "Subaru" },
                { "SUZU",  "Suzuki" },
                { "TESL",  "Tesla" },
                { "TOYO",  "Toyota" },
                { "VOLK",  "Volkswagen" },
                { "VOLV",  "Volvo" },
            };

        /// <summary>
        /// Expands a truncated maker name (e.g. "HOND") to its full form (e.g. "Honda").
        /// Returns the original value unchanged when no match is found.
        /// </summary>
        private static string ExtendMakerName(string maker)
        {
            if (string.IsNullOrEmpty(maker)) return maker;
            return MakerAliases.TryGetValue(maker.Trim(), out string full) ? full : maker;
        }

        private static bool TryExtractField(string line, string fieldName, out string value)
        {
            // Match "FieldName: value" with optional whitespace around the colon
            var match = Regex.Match(line, $@"^{Regex.Escape(fieldName)}\s*:\s*(.+)$", RegexOptions.IgnoreCase);
            if (match.Success)
            {
                value = match.Groups[1].Value.Trim();
                return true;
            }
            value = null;
            return false;
        }
    }
}
