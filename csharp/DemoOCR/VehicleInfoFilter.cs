using System;
using System.Text.RegularExpressions;

namespace DemoOCR
{
    public class VehicleInfo
    {
        public string Year { get; set; }
        public string Make { get; set; }
        public string Color { get; set; }
        public string Expire { get; set; }
        public string Insurance { get; set; }

        public bool IsEmpty =>
            string.IsNullOrEmpty(Year) &&
            string.IsNullOrEmpty(Make) &&
            string.IsNullOrEmpty(Color) &&
            string.IsNullOrEmpty(Expire) &&
            string.IsNullOrEmpty(Insurance);

        public bool IsComplete =>
            !string.IsNullOrEmpty(Year) &&
            !string.IsNullOrEmpty(Make) &&
            !string.IsNullOrEmpty(Color) &&
            !string.IsNullOrEmpty(Expire) &&
            !string.IsNullOrEmpty(Insurance);

        public void MergeFrom(VehicleInfo other)
        {
            if (string.IsNullOrEmpty(Year)      && !string.IsNullOrEmpty(other.Year))      Year      = other.Year;
            if (string.IsNullOrEmpty(Make)      && !string.IsNullOrEmpty(other.Make))      Make      = other.Make;
            if (string.IsNullOrEmpty(Color)     && !string.IsNullOrEmpty(other.Color))     Color     = other.Color;
            if (string.IsNullOrEmpty(Expire)    && !string.IsNullOrEmpty(other.Expire))    Expire    = other.Expire;
            if (string.IsNullOrEmpty(Insurance) && !string.IsNullOrEmpty(other.Insurance)) Insurance = other.Insurance;
        }

        public override string ToString()
        {
            if (IsEmpty) return string.Empty;

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("=== Vehicle Info ===");
            if (!string.IsNullOrEmpty(Year))      sb.AppendLine($"Year:      {Year}");
            if (!string.IsNullOrEmpty(Make))      sb.AppendLine($"Make:      {Make}");
            if (!string.IsNullOrEmpty(Color))     sb.AppendLine($"Color:     {Color}");
            if (!string.IsNullOrEmpty(Expire))    sb.AppendLine($"Expire:    {Expire}");
            if (!string.IsNullOrEmpty(Insurance)) sb.AppendLine($"Insurance: {Insurance}");

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

            string[] lines = ocrText.Split(LineBreaks, StringSplitOptions.None);

            // --- VEHICLE INFO section ---
            if (ocrText.Contains("VEHICLE INFO", StringComparison.OrdinalIgnoreCase))
            {
                int startIndex = -1;
                for (int i = 0; i < lines.Length; i++)
                {
                    if (lines[i].Contains("VEHICLE INFO", StringComparison.OrdinalIgnoreCase))
                    {
                        startIndex = i + 1;
                        break;
                    }
                }

                if (startIndex >= 0)
                {
                    string[] vehicleFields = { "Year", "Make", "Color" };
                    string[] fieldLines = ReassembleLines(lines, startIndex, vehicleFields);

                    foreach (string line in fieldLines)
                    {
                        string trimmed = line.Trim();
                        if (string.IsNullOrEmpty(trimmed)) continue;

                        if (TryExtractField(trimmed, "Year", out string year))
                            info.Year = year;
                        else if (TryExtractField(trimmed, "Make", out string make))
                            info.Make = ExtendMakerName(make);
                        else if (TryExtractField(trimmed, "Color", out string color))
                            info.Color = color;
                    }
                }
            }

            // --- Expiration Date (registration info section) ---
            // OCR may scatter the date across several lines with noise words in between, e.g.:
            //   "Expiration Date: 08/31/2"  "Recent"  "025"  "Last Updated: ..."
            // Strategy: grab everything between "Expiration Date:" and "Last Update", then
            // strip non-digit/slash characters to reconstruct the raw date string.
            string joined = JoinLines(lines);
            var expireRangeMatch = Regex.Match(joined,
                @"Expiration\s+Date\s*:\s*(.+?)\s*Last\s+Updat",
                RegexOptions.IgnoreCase | RegexOptions.Singleline);
            if (expireRangeMatch.Success)
            {
                // Keep only digits and slashes, collapse to a clean date token
                string raw = Regex.Replace(expireRangeMatch.Groups[1].Value, @"[^\d/]", "");
                // raw should now be like "08/31/2025"
                var dateMatch = Regex.Match(raw, @"(\d{2})/(\d{2})/(\d{4})");
                if (dateMatch.Success)
                {
                    int month = int.Parse(dateMatch.Groups[1].Value);
                    int year  = int.Parse(dateMatch.Groups[3].Value);
                    info.Expire = $"{MonthName(month)}, {year}";
                }
            }

            // --- Insurance (INSURANCE COVERAGE section) ---
            // Format: "Response code: Confirmed"
            var insuranceMatch = Regex.Match(ocrText, @"Response\s+[Cc]ode\s*:\s*(\w+)", RegexOptions.IgnoreCase);
            if (insuranceMatch.Success)
            {
                info.Insurance = insuranceMatch.Groups[1].Value.ToUpperInvariant();
            }

            return info;
        }

        private static string JoinLines(string[] lines)
        {
            // Join all lines with a space so fields split across lines can be matched
            return string.Join(" ", lines);
        }

        private static string MonthName(int month)
        {
            return month switch
            {
                1  => "January",
                2  => "February",
                3  => "March",
                4  => "April",
                5  => "May",
                6  => "June",
                7  => "July",
                8  => "August",
                9  => "September",
                10 => "October",
                11 => "November",
                12 => "December",
                _  => month.ToString()
            };
        }

        /// <summary>
        /// Reassembles lines, merging broken field lines such as:
        ///   "Mode"  followed by  "l: FIS"  → "Model: FIS"
        /// The merge heuristic: if a line looks like the tail of a known field name
        /// (i.e. it does not itself start with a known field prefix and the concatenation
        /// of it with the previous line produces a recognized field pattern), merge them.
        /// </summary>
        private static string[] ReassembleLines(string[] lines, int startIndex, string[] knownFields)
        {
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
