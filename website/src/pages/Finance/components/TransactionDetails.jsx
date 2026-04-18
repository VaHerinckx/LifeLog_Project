import React from 'react';
import { X, Calendar, Building, FileText, TrendingUp, Clock } from 'lucide-react';
import './TransactionDetails.css';

/**
 * Get the appropriate emoji for a transaction type
 * @param {string} type - The transaction type
 * @returns {string} Emoji string
 */
const getTransactionEmoji = (type) => {
  const emojiMap = {
    'expense': '💸',
    'income': '💰',
    'incoming_transfer': '🔄',
    'outgoing_transfer': '🔄',
  };
  return emojiMap[type] || '💵';
};

const TransactionDetails = ({ transaction, onClose }) => {
  if (!transaction) return null;

  const transactionEmoji = getTransactionEmoji(transaction.transaction_type);

  // Helper function to format date with time
  const formatDateTime = (dateStr) => {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) return dateStr;

    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Helper function to format amount
  const formatAmount = (amount) => {
    const num = parseFloat(amount) || 0;
    return `€${Math.abs(num).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })}`;
  };

  // Helper function to get transaction type label
  const getTypeLabel = (type) => {
    switch (type) {
      case 'expense':
        return 'Expense';
      case 'income':
        return 'Income';
      case 'incoming_transfer':
        return 'Transfer In';
      case 'outgoing_transfer':
        return 'Transfer Out';
      default:
        return type;
    }
  };

  // Helper function to get transaction type class
  const getTypeClass = (type) => {
    switch (type) {
      case 'expense':
        return 'type-expense';
      case 'income':
        return 'type-income';
      case 'incoming_transfer':
        return 'type-transfer-in';
      case 'outgoing_transfer':
        return 'type-transfer-out';
      default:
        return 'type-default';
    }
  };

  // Helper function to get amount class
  const getAmountClass = () => {
    switch (transaction.transaction_type) {
      case 'expense':
      case 'outgoing_transfer':
        return 'amount-negative';
      case 'income':
      case 'incoming_transfer':
        return 'amount-positive';
      default:
        return '';
    }
  };

  // Check if transaction was converted from another currency
  const isConverted = transaction.currency && transaction.currency !== 'EUR';
  const originalAmount = transaction.amount ? parseFloat(transaction.amount) : null;

  return (
    <div className="transaction-details-overlay" onClick={onClose}>
      <div className="transaction-details-modal" onClick={(e) => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>
          <X size={24} />
        </button>

        <div className="transaction-details-content">
          <div className="transaction-details-header">
            <div className="header-icon-container">
              <span className="transaction-emoji-large">{transactionEmoji}</span>
            </div>
            <div className="header-info">
              <h2>{transaction.category}</h2>
              {transaction.subcategory && <h3>{transaction.subcategory}</h3>}
              <div className={`detail-amount ${getAmountClass()}`}>
                {formatAmount(transaction.corrected_eur)}
              </div>
              <span className={`transaction-type-badge ${getTypeClass(transaction.transaction_type)}`}>
                {getTypeLabel(transaction.transaction_type)}
              </span>
            </div>
          </div>

          <div className="transaction-details-body">
            {/* Basic Transaction Info */}
            <div className="details-section">
              <h4>Transaction Details</h4>

              <div className="detail-item">
                <Calendar size={18} />
                <div className="detail-content">
                  <label>Date & Time:</label>
                  <span>{formatDateTime(transaction.date)}</span>
                </div>
              </div>

              <div className="detail-item">
                <Building size={18} />
                <div className="detail-content">
                  <label>Account:</label>
                  <span>{transaction.accounts}</span>
                </div>
              </div>

              {transaction.note && (
                <div className="detail-item">
                  <FileText size={18} />
                  <div className="detail-content">
                    <label>Note:</label>
                    <span>{transaction.note}</span>
                  </div>
                </div>
              )}

              {transaction.description && (
                <div className="detail-item">
                  <FileText size={18} />
                  <div className="detail-content">
                    <label>Description:</label>
                    <span>{transaction.description}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Financial Details */}
            {isConverted && (
              <div className="details-section financial-section">
                <h4>Financial Information</h4>

                <div className="detail-item">
                  <TrendingUp size={18} />
                  <div className="detail-content">
                    <label>Original Amount:</label>
                    <span>
                      {originalAmount !== null ? originalAmount.toLocaleString('en-US', {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                      }) : 'N/A'} {transaction.currency}
                    </span>
                  </div>
                </div>

                <div className="detail-item">
                  <DollarSign size={18} />
                  <div className="detail-content">
                    <label>EUR Equivalent:</label>
                    <span>{formatAmount(transaction.corrected_eur)}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Movement Info */}
            <div className="details-section">
              <h4>Movement</h4>

              <div className="detail-item">
                <TrendingUp size={18} />
                <div className="detail-content">
                  <label>Net Movement:</label>
                  <span className={parseFloat(transaction.movement) >= 0 ? 'amount-positive' : 'amount-negative'}>
                    {parseFloat(transaction.movement) >= 0 ? '+' : ''}
                    {formatAmount(transaction.movement)}
                  </span>
                </div>
              </div>
            </div>

            {/* Temporal Context */}
            {(transaction.year_week || transaction.year_month) && (
              <div className="details-section">
                <h4>Time Context</h4>

                {transaction.year_week && (
                  <div className="detail-item">
                    <Clock size={18} />
                    <div className="detail-content">
                      <label>Week:</label>
                      <span>{transaction.year_week}</span>
                    </div>
                  </div>
                )}

                {transaction.year_month && (
                  <div className="detail-item">
                    <Calendar size={18} />
                    <div className="detail-content">
                      <label>Month:</label>
                      <span>{transaction.year_month}</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TransactionDetails;
