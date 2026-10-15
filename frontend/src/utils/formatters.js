export function formatDate(isoString) {
  if (!isoString) return '—';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return String(isoString);
  }
}

export function formatTimeOnly(isoString) {
  if (!isoString) return '—';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return String(isoString);
  }
}

export function formatNumber(num) {
  if (num === null || num === undefined) return '0';
  return Number(num).toLocaleString();
}

export function formatPercent(num, decimals = 1) {
  if (num === null || num === undefined || isNaN(num)) return '0.0%';
  return `${Number(num).toFixed(decimals)}%`;
}

export function formatMs(ms, decimals = 2) {
  if (ms === null || ms === undefined || isNaN(ms)) return '0.00 ms';
  return `${Number(ms).toFixed(decimals)} ms`;
}

export function truncateId(id, startChars = 8, endChars = 4) {
  if (!id) return '';
  if (id.length <= startChars + endChars) return id;
  return `${id.substring(0, startChars)}...${id.substring(id.length - endChars)}`;
}
