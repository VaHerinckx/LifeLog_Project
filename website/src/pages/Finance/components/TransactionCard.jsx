import PropTypes from 'prop-types';
import { Building, Calendar } from 'lucide-react';
import { formatDate } from '../../../utils';
import './TransactionCard.css';

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

/**
 * TransactionCard - Displays transaction information in grid or list view
 *
 * Self-contained component that adapts its layout based on viewMode prop.
 * All styling is contained in TransactionCard.css.
 *
 * Grid view: Vertical layout with emoji badge in top right
 * List view: Horizontal layout with emoji badge in top right
 *
 * @param {Object} transaction - Transaction data object
 * @param {string} viewMode - Display mode ('grid' | 'list')
 * @param {Function} onClick - Callback when card is clicked
 */
const TransactionCard = ({ transaction, viewMode = 'grid', onClick }) => {
  const cardClass = `transaction-card transaction-card--${viewMode}`;
  const transactionEmoji = getTransactionEmoji(transaction.transaction_type);

  // Helper function to get transaction type badge class
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

  // Helper function to format transaction type label
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

  // Helper function to format amount with proper sign and color
  const formatAmount = () => {
    const amount = parseFloat(transaction.corrected_eur) || 0;
    const formatted = `€${Math.abs(amount).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    })}`;

    return formatted;
  };

  // Helper function to get amount class based on transaction type
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

  // Helper function to truncate note text
  const truncateNote = (text, maxLength = 50) => {
    if (!text) return '';
    return text.length > maxLength ? `${text.substring(0, maxLength)}...` : text;
  };

  // Show currency badge if not EUR
  const showCurrencyBadge = transaction.currency && transaction.currency !== 'EUR';

  // List view - horizontal layout
  if (viewMode === 'list') {
    return (
      <div className={cardClass} onClick={() => onClick(transaction)}>
        <div className="transaction-emoji-badge">{transactionEmoji}</div>

        <div className="transaction-info">
          <div className="transaction-header">
            <h3 className="transaction-category">{transaction.category}</h3>
            <span className={`transaction-amount ${getAmountClass()}`}>
              {formatAmount()}
            </span>
          </div>

          {transaction.subcategory && (
            <p className="transaction-subcategory">{transaction.subcategory}</p>
          )}

          {transaction.note && (
            <p className="transaction-note">{transaction.note}</p>
          )}

          <div className="transaction-meta">
            <div className="meta-item">
              <Calendar size={14} />
              <span>{formatDate(transaction.date)}</span>
            </div>
            <div className="meta-item">
              <Building size={14} />
              <span>{transaction.Accounts || transaction.accounts}</span>
            </div>
          </div>
        </div>

        <div className="transaction-badges">
          <span className={`transaction-type-badge ${getTypeClass(transaction.transaction_type)}`}>
            {getTypeLabel(transaction.transaction_type)}
          </span>
          {showCurrencyBadge && (
            <span className="currency-badge">{transaction.currency}</span>
          )}
        </div>
      </div>
    );
  }

  // Grid view - vertical layout (default)
  return (
    <div className={cardClass} onClick={() => onClick(transaction)}>
      <div className="transaction-emoji-badge">{transactionEmoji}</div>

      <div className="transaction-info">
        <h3 className="transaction-category" title={transaction.category}>
          {transaction.category}
        </h3>

        {transaction.subcategory && (
          <p className="transaction-subcategory" title={transaction.subcategory}>
            {transaction.subcategory}
          </p>
        )}

        <div className={`transaction-amount ${getAmountClass()}`}>
          {formatAmount()}
        </div>

        {transaction.note && (
          <p className="transaction-note" title={transaction.note}>
            {truncateNote(transaction.note, 40)}
          </p>
        )}

        <div className="transaction-meta">
          <div className="meta-item">
            <Calendar size={14} />
            <span>{formatDate(transaction.date)}</span>
          </div>
          <div className="meta-item">
            <Building size={14} />
            <span className="account-name">{transaction.Accounts || transaction.accounts}</span>
          </div>
        </div>

        <div className="transaction-badges">
          <span className={`transaction-type-badge ${getTypeClass(transaction.transaction_type)}`}>
            {getTypeLabel(transaction.transaction_type)}
          </span>
          {showCurrencyBadge && (
            <span className="currency-badge">{transaction.currency}</span>
          )}
        </div>
      </div>
    </div>
  );
};

TransactionCard.propTypes = {
  transaction: PropTypes.shape({
    date: PropTypes.oneOfType([PropTypes.string, PropTypes.instanceOf(Date)]).isRequired,
    accounts: PropTypes.string.isRequired,
    category: PropTypes.string.isRequired,
    subcategory: PropTypes.string,
    note: PropTypes.string,
    corrected_eur: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    currency: PropTypes.string,
    transaction_type: PropTypes.string.isRequired,
  }).isRequired,
  viewMode: PropTypes.oneOf(['grid', 'list']),
  onClick: PropTypes.func.isRequired,
};

export default TransactionCard;
