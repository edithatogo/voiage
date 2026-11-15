// Copyright 2026. Licensed under the Apache License, Version 2.0.
using System.IO.Compression;
using System.IO.Packaging;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Xml;
using System.Xml.Linq;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Validation;
using SharpZipFile = ICSharpCode.SharpZipLib.Zip.ZipFile;

internal static class Program
{
    private const string RelationshipNamespace =
        "http://schemas.openxmlformats.org/package/2006/relationships";

    private static int Main(string[] args)
    {
        if (args.Length < 3 || args[0] != "--report")
        {
            Console.Error.WriteLine("Usage: DocxGate --report NEW_RECEIPT.json FILE.docx [FILE.docx ...]");
            return 2;
        }

        try
        {
            string reportPath = Path.GetFullPath(args[1]);
            string[] files = args.Skip(2).Select(Path.GetFullPath).ToArray();
            if (File.Exists(reportPath) || files.Contains(reportPath, StringComparer.OrdinalIgnoreCase))
            {
                throw new IOException("The receipt must be a new path distinct from every input.");
            }

            var results = files.Select(ValidateFile).ToArray();
            bool passed = results.All(result => result.Passed);
            string assemblyPath = Assembly.GetExecutingAssembly().Location;
            var receipt = new
            {
                gate = "manuscript-docx-structure-v1",
                utc = DateTimeOffset.UtcNow,
                validator = typeof(OpenXmlValidator).Assembly.GetName().Version?.ToString(),
                target = "Microsoft365",
                runtime = RuntimeInformation.FrameworkDescription,
                tool = new
                {
                    assembly_sha256 = HashFile(assemblyPath),
                    openxml_sha256 = HashFile(typeof(OpenXmlValidator).Assembly.Location),
                    zip_library = typeof(SharpZipFile).Assembly.GetName().Version?.ToString(),
                    zip_library_sha256 = HashFile(typeof(SharpZipFile).Assembly.Location),
                    packaging_sha256 = HashFile(typeof(PackUriHelper).Assembly.Location),
                    os = RuntimeInformation.OSDescription,
                },
                passed,
                files = results,
                limits = new[]
                {
                    "The structural gate does not establish scientific or author acceptance.",
                    "Fidelity, rendering, privacy and applicable native Word checks are separate.",
                    "A change to final file bytes invalidates this receipt.",
                    "External relationships are not fetched.",
                },
            };

            Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
            using var output = new FileStream(reportPath, FileMode.CreateNew, FileAccess.Write);
            JsonSerializer.Serialize(output, receipt, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            });
            return passed ? 0 : 1;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"Cannot run DOCX validation: {error.Message}");
            return 2;
        }
    }

    private static FileResult ValidateFile(string path)
    {
        var issues = new List<object>();
        string? sha256 = null;
        int schemaErrors = 0;

        try
        {
            if (!path.EndsWith(".docx", StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException("Expected a .docx file.");
            }

            using var input = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            sha256 = HashStream(input);
            input.Position = 0;
            ValidateZip(input);
            input.Position = 0;

            using var document = WordprocessingDocument.Open(input, false);
            if (document.DocumentType != WordprocessingDocumentType.Document)
            {
                throw new InvalidDataException("The package is not a DOCX document.");
            }

            var validator = new OpenXmlValidator(FileFormatVersions.Microsoft365)
            {
                MaxNumberOfErrors = 0,
            };
            foreach (var error in validator.Validate(document))
            {
                schemaErrors++;
                issues.Add(new
                {
                    kind = "openxml",
                    id = error.Id,
                    part = error.Part?.Uri.ToString(),
                    xpath = error.Path?.XPath,
                    description = error.Description,
                });
            }
        }
        catch (Exception error)
        {
            issues.Add(new { kind = "package_or_runtime", description = error.Message });
        }

        if (sha256 is not null)
        {
            try
            {
                if (HashFile(path) != sha256)
                {
                    issues.Add(new { kind = "input_changed", description = "Input changed during validation." });
                }
            }
            catch (Exception error)
            {
                issues.Add(new { kind = "input_changed", description = error.Message });
            }
        }

        bool passed = issues.Count == 0;
        Console.WriteLine($"{(passed ? "PASS" : "FAIL")} {Path.GetFileName(path)} ({issues.Count} issue(s))");
        return new FileResult(path, sha256, passed, schemaErrors, issues);
    }

    private static void ValidateZip(Stream input)
    {
        using (var zip = new SharpZipFile(input) { IsStreamOwner = false })
        {
            if (!zip.TestArchive(testData: true))
            {
                throw new InvalidDataException("ZIP header or CRC validation failed.");
            }
        }
        input.Position = 0;
        using var archive = new ZipArchive(input, ZipArchiveMode.Read, leaveOpen: true);
        var names = archive.Entries.Select(entry => entry.FullName).ToHashSet(StringComparer.Ordinal);
        if (names.Count != archive.Entries.Count)
        {
            throw new InvalidDataException("The package contains duplicate ZIP part names.");
        }

        foreach (string required in new[] { "[Content_Types].xml", "_rels/.rels", "word/document.xml" })
        {
            if (!names.Contains(required))
            {
                throw new InvalidDataException($"Missing required part: {required}");
            }
        }

        foreach (var entry in archive.Entries)
        {
            if (!entry.FullName.EndsWith(".xml", StringComparison.Ordinal)
                && !entry.FullName.EndsWith(".rels", StringComparison.Ordinal))
            {
                continue;
            }

            using Stream xmlStream = entry.Open();
            using var reader = XmlReader.Create(xmlStream, new XmlReaderSettings
            {
                DtdProcessing = DtdProcessing.Prohibit,
                XmlResolver = null,
            });
            XDocument xml = XDocument.Load(reader);
            if (entry.FullName.EndsWith(".rels", StringComparison.Ordinal))
            {
                ValidateRelationships(entry.FullName, xml, names);
            }
        }
    }

    private static void ValidateRelationships(string name, XDocument xml, HashSet<string> names)
    {
        XName rootName = XName.Get("Relationships", RelationshipNamespace);
        if (xml.Root?.Name != rootName)
        {
            throw new InvalidDataException($"Invalid relationship root: {name}");
        }

        var relationshipPart = new Uri("/" + name, UriKind.Relative);
        Uri source = PackUriHelper.GetSourcePartUriFromRelationshipPartUri(relationshipPart);
        if (source.OriginalString != "/" && !names.Contains(PartName(source)))
        {
            throw new InvalidDataException($"Missing relationship source: {PartName(source)}");
        }

        foreach (XElement relationship in xml.Root.Elements(XName.Get("Relationship", RelationshipNamespace)))
        {
            if ((string?)relationship.Attribute("TargetMode") == "External")
            {
                continue;
            }

            string target = (string?)relationship.Attribute("Target")
                ?? throw new InvalidDataException($"Relationship in {name} lacks a target.");
            var targetUri = new Uri(target, UriKind.RelativeOrAbsolute);
            if (targetUri.IsAbsoluteUri)
            {
                throw new InvalidDataException($"Internal relationship has an absolute target: {target}");
            }

            Uri resolved = PackUriHelper.ResolvePartUri(source, targetUri);
            if (!names.Contains(PartName(resolved)))
            {
                throw new InvalidDataException($"Missing relationship target: {PartName(resolved)}");
            }
        }
    }

    private static string PartName(Uri uri) => uri.OriginalString.TrimStart('/');

    private static string HashFile(string path)
    {
        using var input = File.OpenRead(path);
        return HashStream(input);
    }

    private static string HashStream(Stream input) => Convert.ToHexString(SHA256.HashData(input)).ToLowerInvariant();

    private sealed record FileResult(
        string Path,
        string? Sha256,
        bool Passed,
        int SchemaErrors,
        List<object> Issues);
}
