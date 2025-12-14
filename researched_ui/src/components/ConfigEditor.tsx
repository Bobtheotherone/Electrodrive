import React, { useEffect, useMemo, useRef, useState } from "react";

export interface ConfigEditorProps {
  label?: string;
  value: unknown; // initial object
  schema?: any; // JSON Schema-ish
  required?: boolean;
  height?: number; // px
  onChange: (
    value: unknown | null,
    meta: {
      text: string;
      parseError?: string;
      validationErrors?: string[];
      isValid: boolean;
    }
  ) => void;
}

type ValidationResult = {
  errors: string[];
  schemaError?: string;
};

function safeJsonStringify(value: unknown, space = 2): string {
  try {
    return JSON.stringify(value ?? null, null, space);
  } catch {
    // Cycles or unsupported types: fall back to string.
    return String(value);
  }
}

function looksEmpty(text: string): boolean {
  return text.trim().length === 0;
}

/**
 * Minimal JSON Schema-ish validator (no deps) supporting:
 * - type
 * - required
 * - properties
 * - enum
 * - minimum / maximum for numbers
 * - items for arrays (best effort)
 */
function validateMinimal(schema: any, value: unknown): ValidationResult {
  const errors: string[] = [];
  const maxDepth = 12;

  const typeMatches = (t: string, v: unknown): boolean => {
    if (t === "null") return v === null;
    if (t === "array") return Array.isArray(v);
    if (t === "object") return v !== null && typeof v === "object" && !Array.isArray(v);
    if (t === "string") return typeof v === "string";
    if (t === "number") return typeof v === "number" && Number.isFinite(v);
    if (t === "integer") return typeof v === "number" && Number.isFinite(v) && Number.isInteger(v);
    if (t === "boolean") return typeof v === "boolean";
    return true; // unknown type: don't fail
  };

  const fmtPath = (path: string): string => path || "$";

  const walk = (sch: any, v: unknown, path: string, depth: number) => {
    if (!sch || depth > maxDepth) return;

    // enum
    if (Array.isArray(sch.enum)) {
      const ok = sch.enum.some((e: unknown) => Object.is(e, v));
      if (!ok) errors.push(`${fmtPath(path)} must be one of ${safeJsonStringify(sch.enum, 0)}`);
    }

    // type
    if (sch.type) {
      const allowedTypes: string[] = Array.isArray(sch.type) ? sch.type : [sch.type];
      const ok = allowedTypes.some((t) => typeof t === "string" && typeMatches(t, v));
      if (!ok) errors.push(`${fmtPath(path)} must be of type ${allowedTypes.join(" | ")}`);
    }

    // number bounds
    if (typeof v === "number" && Number.isFinite(v)) {
      if (typeof sch.minimum === "number" && v < sch.minimum) {
        errors.push(`${fmtPath(path)} must be >= ${sch.minimum}`);
      }
      if (typeof sch.maximum === "number" && v > sch.maximum) {
        errors.push(`${fmtPath(path)} must be <= ${sch.maximum}`);
      }
    }

    // object: required/properties
    if (sch.type === "object" || (sch.properties && typeof sch.properties === "object")) {
      if (v === null || typeof v !== "object" || Array.isArray(v)) {
        // If schema says object but value isn't, type error already handled above.
        return;
      }
      const obj = v as Record<string, unknown>;

      if (Array.isArray(sch.required)) {
        for (const key of sch.required) {
          if (typeof key === "string" && !(key in obj)) {
            errors.push(`${fmtPath(path)}.${key} is required`);
          }
        }
      }
      if (sch.properties && typeof sch.properties === "object") {
        for (const [k, child] of Object.entries(sch.properties)) {
          if (k in obj) walk(child, obj[k], path ? `${path}.${k}` : k, depth + 1);
        }
      }
    }

    // array: items
    if (sch.type === "array" && Array.isArray(v) && sch.items) {
      const arr = v as unknown[];
      for (let i = 0; i < Math.min(arr.length, 200); i++) {
        walk(sch.items, arr[i], `${fmtPath(path)}[${i}]`, depth + 1);
      }
    }
  };

  walk(schema, value, "$", 0);
  return { errors };
}

async function tryLoadAjv(): Promise<any | null> {
  try {
    // Avoid a hard dependency: do not reference `import("ajv")` directly so TS won't require module types.
    const importer = new Function("m", "return import(m);") as (m: string) => Promise<any>;
    const mod = await importer("ajv");
    return mod?.default ?? mod ?? null;
  } catch {
    return null;
  }
}

function normalizeAjvErrors(errs: any): string[] {
  if (!Array.isArray(errs)) return ["Schema validation failed."];
  return errs
    .map((e) => {
      const inst = typeof e.instancePath === "string" ? e.instancePath : "";
      const kw = typeof e.keyword === "string" ? e.keyword : "invalid";
      const msg = typeof e.message === "string" ? e.message : "invalid";
      const loc = inst ? `$${inst}` : "$";
      const params = e.params ? ` (${safeJsonStringify(e.params, 0)})` : "";
      return `${loc}: ${kw} ${msg}${params}`;
    })
    .filter(Boolean);
}

export default function ConfigEditor(props: ConfigEditorProps) {
  const { label = "Config (JSON)", value, schema, required = false, height = 220, onChange } = props;

  const [text, setText] = useState<string>(() => {
    if (value === undefined || value === null) return "";
    const s = safeJsonStringify(value, 2);
    return s === "null" ? "" : s;
  });

  const [touched, setTouched] = useState<boolean>(false);

  // Optional Ajv support
  const ajvRef = useRef<any | null>(null);
  const [ajvState, setAjvState] = useState<"loading" | "available" | "unavailable">("loading");

  useEffect(() => {
    let mounted = true;
    (async () => {
      const AjvCtor = await tryLoadAjv();
      if (!mounted) return;

      if (!AjvCtor) {
        ajvRef.current = null;
        setAjvState("unavailable");
        return;
      }

      try {
        ajvRef.current = new AjvCtor({ allErrors: true, strict: false });
        setAjvState("available");
      } catch {
        ajvRef.current = null;
        setAjvState("unavailable");
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  // Keep textarea in sync with external value only until user edits.
  useEffect(() => {
    if (touched) return;
    const next = value === undefined || value === null ? "" : safeJsonStringify(value, 2);
    setText(next === "null" ? "" : next);
  }, [value, touched]);

  // Stable schema fingerprint so we don't treat “new object identity” as a schema change.
  const schemaKey = useMemo(() => {
    if (!schema) return "";
    return safeJsonStringify(schema, 0);
  }, [schema]);

  const { compiledValidator, schemaCompileError } = useMemo(() => {
    if (!schema) return { compiledValidator: null as any, schemaCompileError: undefined as string | undefined };
    if (ajvState !== "available" || !ajvRef.current) return { compiledValidator: null as any, schemaCompileError: undefined as string | undefined };

    try {
      const fn = ajvRef.current.compile(schema);
      return { compiledValidator: fn, schemaCompileError: undefined };
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return { compiledValidator: null as any, schemaCompileError: msg };
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schemaKey, ajvState]);

  const computeMeta = (nextText: string) => {
    const metaBase = {
      text: nextText,
      parseError: undefined as string | undefined,
      validationErrors: undefined as string[] | undefined,
      isValid: false,
    };

    if (looksEmpty(nextText)) {
      const isValidEmpty = !required;
      return { parsed: null as unknown | null, meta: { ...metaBase, isValid: isValidEmpty } };
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(nextText) as unknown;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return { parsed: null as unknown | null, meta: { ...metaBase, parseError: msg, isValid: false } };
    }

    // If schema compile error exists, treat as schema validation error but don't block parse.
    if (schemaCompileError) {
      return {
        parsed,
        meta: { ...metaBase, validationErrors: [`Schema error: ${schemaCompileError}`], isValid: false },
      };
    }

    if (schema) {
      // Prefer Ajv if loaded/compiled.
      if (compiledValidator) {
        const ok = Boolean(compiledValidator(parsed));
        if (!ok) {
          const errs = normalizeAjvErrors(compiledValidator.errors);
          return { parsed, meta: { ...metaBase, validationErrors: errs, isValid: false } };
        }
      } else {
        const res = validateMinimal(schema, parsed);
        if (res.schemaError) return { parsed, meta: { ...metaBase, validationErrors: [res.schemaError], isValid: false } };
        if (res.errors.length > 0) return { parsed, meta: { ...metaBase, validationErrors: res.errors, isValid: false } };
      }
    }

    return { parsed, meta: { ...metaBase, isValid: true } };
  };

  const computed = useMemo(() => computeMeta(text), [text, schemaKey, ajvState, schemaCompileError, compiledValidator, required]);

  // Emit changes to parent (deduped) so inline schemas and parent re-renders don't cause churn loops.
  const lastEmitKey = useRef<string>("");
  const lastEmitOnChange = useRef<ConfigEditorProps["onChange"] | null>(null);

  useEffect(() => {
    const ve = computed.meta.validationErrors?.join("\n") ?? "";
    const key = `${computed.meta.text}\n${computed.meta.isValid}\n${computed.meta.parseError ?? ""}\n${ve}`;

    if (lastEmitKey.current === key && lastEmitOnChange.current === onChange) return;

    lastEmitKey.current = key;
    lastEmitOnChange.current = onChange;
    onChange(computed.parsed, computed.meta);
  }, [computed.parsed, computed.meta, onChange]);

  const id = React.useId();
  const helpId = `${id}-help`;
  const errId = `${id}-err`;

  const onFormat = () => {
    const { parsed, meta } = computeMeta(text);
    if (meta.parseError) return;
    if (parsed === null) return;
    setText(safeJsonStringify(parsed, 2));
    setTouched(true);
  };

  const onReset = () => {
    const next = value === undefined || value === null ? "" : safeJsonStringify(value, 2);
    setText(next === "null" ? "" : next);
    setTouched(false);
  };

  const onValidateNow = () => {
    // Forces touched so errors become visible in UI patterns.
    setTouched(true);
    const { parsed, meta } = computeMeta(text);
    onChange(parsed, meta);
  };

  const meta = computed.meta;
  const hasErrors = Boolean(meta.parseError) || (meta.validationErrors && meta.validationErrors.length > 0);

  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, flexWrap: "wrap" }}>
        <label htmlFor={id} style={{ fontWeight: 600 }}>
          {label}
          {required ? <span aria-hidden="true"> *</span> : null}
        </label>

        <div style={{ display: "flex", gap: 8, marginLeft: "auto", flexWrap: "wrap" }}>
          <button type="button" onClick={onFormat} disabled={!!meta.parseError || looksEmpty(text)} style={{ padding: "4px 8px" }}>
            Format JSON
          </button>
          <button type="button" onClick={onValidateNow} style={{ padding: "4px 8px" }}>
            Validate
          </button>
          <button type="button" onClick={onReset} style={{ padding: "4px 8px" }}>
            Reset
          </button>
        </div>
      </div>

      <div id={helpId} style={{ fontSize: 12, color: "#444" }}>
        {schema ? (
          <span>
            Validation: {ajvState === "available" ? "Ajv" : ajvState === "loading" ? "Built-in (Ajv loading…)" : "Built-in"}{" "}
            {schemaCompileError ? "(schema compile error)" : ""}
          </span>
        ) : (
          <span>No schema provided; only JSON parsing is checked.</span>
        )}
      </div>

      <textarea
        id={id}
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setTouched(true);
        }}
        spellCheck={false}
        aria-describedby={hasErrors ? `${helpId} ${errId}` : helpId}
        style={{
          width: "100%",
          height,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
          fontSize: 12,
          padding: 10,
          borderRadius: 6,
          border: hasErrors ? "1px solid #b00020" : "1px solid #bbb",
          background: "#fff",
        }}
        placeholder={required ? `Enter JSON (required)` : `Enter JSON (optional)`}
      />

      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span
          aria-live="polite"
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: meta.isValid ? "#0b6" : hasErrors ? "#b00020" : "#444",
          }}
        >
          {meta.isValid ? "Valid" : required && looksEmpty(text) ? "Required" : hasErrors ? "Invalid" : "—"}
        </span>

        {schemaCompileError ? (
          <span style={{ fontSize: 12, color: "#b00020" }}>
            Schema error: <code>{schemaCompileError}</code>
          </span>
        ) : null}
      </div>

      {touched && hasErrors ? (
        <div id={errId} role="alert" style={{ border: "1px solid #b00020", borderRadius: 6, padding: 10, background: "#fff5f6" }}>
          {meta.parseError ? (
            <div style={{ fontSize: 12, color: "#b00020" }}>
              <strong>Parse error:</strong> <span>{meta.parseError}</span>
            </div>
          ) : null}

          {meta.validationErrors && meta.validationErrors.length > 0 ? (
            <div style={{ marginTop: meta.parseError ? 8 : 0 }}>
              <div style={{ fontSize: 12, color: "#b00020", fontWeight: 600 }}>Validation errors:</div>
              <ul style={{ margin: "6px 0 0 18px", fontSize: 12, color: "#b00020" }}>
                {meta.validationErrors.slice(0, 50).map((e, idx) => (
                  <li key={idx}>{e}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
